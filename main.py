import argparse
import os
from typing import Any, Dict

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from pkg.data_module import MolecularDataModule
from pkg.transformer import DiT
from pkg.generator import MolecularGenerator, EpochEvaluator


class DiffusionScheduler:
    """Simple linear noise scheduler for diffusion training."""
    
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2):
        self.num_timesteps = num_timesteps
        
        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Add noise to clean data according to diffusion forward process."""
        # Ensure scheduler tensors are on the same device as input tensors
        device = x_0.device
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        
        sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise


class MolecularDiffusionModel(L.LightningModule):
    """Lightning module for molecular diffusion model training."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        num_layers: int = 12,
        nhead: int = 6,
        mlp_ratio: float = 4.0,
        dreams_emb_dim: int = 1024,
        max_length: int = 500,
        learning_rate: float = 1e-4,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        vocab_file: str = "data/vocab.txt",
        eval_every_n_epochs: int = 5,
        num_eval_samples: int = 8,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize DiT model
        self.model = DiT(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            mlp_ratio=mlp_ratio,
            dreams_emb_dim=dreams_emb_dim,
            max_length=max_length,
        )
        
        # Initialize diffusion scheduler
        self.scheduler = DiffusionScheduler(
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end
        )
        
        self.learning_rate = learning_rate
        self.vocab_file = vocab_file
        self.eval_every_n_epochs = eval_every_n_epochs
        self.num_eval_samples = num_eval_samples
        
        # Initialize generator and evaluator (will be setup in on_train_start)
        self.generator = None
        self.evaluator = None
        
        # Move scheduler tensors to device when model is moved
        self.register_buffer('_dummy', torch.tensor(0.0))
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")
    
    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")
    
    def _shared_step(self, batch, stage):
        dreams_embedding = batch['dreams_embedding']  # Clean dreams embedding data
        x_0 = batch['tokens']
        x_0_emb = self.model.embed_x(x_0)
        
        batch_size = x_0_emb.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.scheduler.num_timesteps, (batch_size,), device=x_0_emb.device)
        
        # Sample noise
        noise = torch.randn_like(x_0_emb)
        
        # Add noise to clean data
        x_t = self.scheduler.add_noise(x_0_emb, t, noise)
        # Predict the noise
        predicted_score = self.model(
            x_t, 
            t.unsqueeze(1), 
            dreams_embedding,
        )

        # Ensure scheduler tensor is on correct device
        sqrt_alphas_cumprod = self.scheduler.sqrt_alphas_cumprod.to(x_0_emb.device)
        loss = predicted_score + (x_t - x_0_emb) / sqrt_alphas_cumprod[t].reshape(-1, 1, 1)
        loss = torch.mean(loss ** 2)
        
        # Logging
        self.log(f'{stage}_loss', loss, on_step=(stage=='train'), on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.95)
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.learning_rate * 0.1
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
    
    def on_train_start(self) -> None:
        """Setup generator and evaluator when training starts."""
        # Import here to avoid circular imports
        from deepchem.feat.smiles_tokenizer import SmilesTokenizer
        
        # Initialize tokenizer
        tokenizer = SmilesTokenizer(vocab_file=self.vocab_file)
        
        # Initialize generator
        self.generator = MolecularGenerator(
            model=self.model,
            scheduler=self.scheduler,
            tokenizer=tokenizer,
            max_length=self.model.max_length
        )
        
        # Initialize evaluator
        self.evaluator = EpochEvaluator(
            generator=self.generator,
            num_samples=self.num_eval_samples,
            log_interval=self.eval_every_n_epochs
        )
    
    def on_train_epoch_end(self) -> None:
        """Run evaluation after each training epoch."""
        if self.evaluator is not None and hasattr(self.trainer, 'current_epoch'):
            try:
                metrics = self.evaluator.evaluate(self, self.trainer.current_epoch)
                
                # Print some example generated SMILES
                if metrics and self.trainer.current_epoch % self.eval_every_n_epochs == 0:
                    if self.evaluator.evaluation_history:
                        latest_eval = self.evaluator.evaluation_history[-1]
                        examples = latest_eval['examples']
                        
                        print(f"\n=== Epoch {self.trainer.current_epoch} Generation Examples ===")
                        for i, (gen, ref) in enumerate(zip(examples['generated_smiles'], examples['reference_smiles'])):
                            print(f"Generated {i+1}: {gen}")
                            print(f"Reference {i+1}: {ref}")
                            print()
            except Exception as e:
                print(f"Evaluation failed: {e}")
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # Ensure scheduler tensors are on the correct device
        device = self._dummy.device
        for key, value in self.scheduler.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self.scheduler, key, value.to(device))


def main():
    torch.set_float32_matmul_precision("medium")
    
    parser = argparse.ArgumentParser(description="Train molecular diffusion model")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--vocab_file", type=str, default="data/vocab.txt", help="Vocabulary file")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_length", type=int, default=500, help="Maximum sequence length")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers")
    
    # Model parameters
    parser.add_argument("--d_model", type=int, default=384, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--nhead", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--mlp_ratio", type=float, default=4.0, help="MLP expansion ratio")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum epochs")
    parser.add_argument("--num_timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--beta_start", type=float, default=1e-4, help="Diffusion beta start")
    parser.add_argument("--beta_end", type=float, default=4e-2, help="Diffusion beta end")
    
    # Training setup
    parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator type")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    parser.add_argument("--precision", type=str, default="32", help="Training precision")
    parser.add_argument("--log_dir", type=str, default="logs", help="Logging directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    
    # Evaluation parameters
    parser.add_argument("--eval_every_n_epochs", type=int, default=1, help="Evaluate every N epochs")
    parser.add_argument("--num_eval_samples", type=int, default=8, help="Number of samples to generate for evaluation")
    
    # Logging parameters
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="molecular-diffusion", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity/team name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None, help="Wandb tags for the run")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize data module
    data_module = MolecularDataModule(
        data_dir=args.data_dir,
        vocab_file=args.vocab_file,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers
    )
    
    # Get dataset info for model initialization
    dataset_info = data_module.get_dataset_info()
    
    # Initialize model with correct dimensions
    model = MolecularDiffusionModel(
        vocab_size=dataset_info['vocab_size'],
        d_model=args.d_model,
        num_layers=args.num_layers,
        nhead=args.nhead,
        mlp_ratio=args.mlp_ratio,
        dreams_emb_dim=dataset_info['dreams_emb_dim'],
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        num_timesteps=args.num_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        vocab_file=args.vocab_file,
        eval_every_n_epochs=args.eval_every_n_epochs,
        num_eval_samples=args.num_eval_samples
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='diffusion-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        every_n_epochs=1
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Setup logger
    if args.use_wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            tags=args.wandb_tags,
            save_dir=args.log_dir,
            log_model=True,  # Log model checkpoints to wandb
        )
        print(f"Using Weights & Biases logging - Project: {args.wandb_project}")
    else:
        logger = TensorBoardLogger(
            save_dir=args.log_dir,
            name="molecular_diffusion",
            version=None
        )
        print(f"Using TensorBoard logging - Log dir: {args.log_dir}")
    
    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Print dataset and model summary
    print(f"\nDataset Summary:")
    print(f"Dreams embedding dimension: {dataset_info['dreams_emb_dim']}")
    print(f"Vocabulary size: {dataset_info['vocab_size']}")
    print(f"Number of instruments: {dataset_info['num_instruments']}")
    print(f"Number of adducts: {dataset_info['num_adducts']}")
    
    print(f"\nModel Summary:")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Start training
    print(f"\nStarting training...")
    trainer.fit(model, data_module)
    
    # Test the model
    print(f"\nTesting model...")
    trainer.test(model, data_module)
    
    print(f"\nTraining completed! Checkpoints saved to: {args.checkpoint_dir}")
    print(f"Logs saved to: {args.log_dir}")


if __name__ == "__main__":
    main()
