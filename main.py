import argparse
import os
import pickle
import time
from typing import Optional

import equinox as eqx
import jax
import jmp
import optax
import wandb
from tqdm import tqdm
from deepchem.feat.smiles_tokenizer import SmilesTokenizer

from pkg.data_module import MolecularDataModule
from pkg.embedding import Embedding
from pkg.evaluator import MolecularEvaluator
from pkg.generator import MolecularGenerator
from pkg.objective import diffusion_lm_loss
from pkg.scheduler import DiffusionScheduler
from pkg.transformer import SequenceDiT


def loss_fn(model, batch, key, scheduler):
    """Compute loss for a batch using the diffusion objective."""
    tokens = batch['tokens']
    dreams_emb = batch['dreams_embedding']
    
    losses = diffusion_lm_loss(
        key=key,
        denoiser=model,
        tokens=tokens,
        max_t=scheduler.num_timesteps - 1,
        cond_emb=dreams_emb,
        sched=scheduler
    )
    
    return losses

@eqx.filter_jit
@eqx.debug.assert_max_traces(max_traces=2)
def train_step(model, opt_state, batch, key, optimizer, scheduler):
    """Single training step with gradient computation."""
    
    def loss_and_grads(model):
        loss = loss_fn(model, batch, key, scheduler)
        return loss
    
    # Compute gradients
    loss, grads = eqx.filter_value_and_grad(loss_and_grads)(model)
    
    # Apply optimizer update
    updates, new_opt_state = optimizer.update(
        grads,
        opt_state,
        eqx.filter(model, eqx.is_array),
    )
    new_model = eqx.apply_updates(model, updates)
    
    return new_model, new_opt_state, loss


class MolecularDiffusionTrainer:
    """JAX-based trainer for molecular diffusion models."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int = 384,
        num_layers: int = 12,
        num_heads: int = 6,
        hidden_size: int = 1536,  # mlp_ratio * embedding_size
        dreams_emb_size: int = 1024,
        max_length: int = 500,
        learning_rate: float = 1e-4,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 4e-2,
        use_mixed_precision: bool = False,
        eval_generation: bool = True,
        eval_max_batches: int = 5,
        eval_temperature: float = 1.0,
        max_steps_per_epoch: Optional[int] = None,
        tokenizer: Optional[SmilesTokenizer] = None,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(42)
        
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.max_steps_per_epoch = max_steps_per_epoch
        
        # Evaluation parameters
        self.eval_generation = eval_generation
        self.eval_max_batches = eval_max_batches
        self.eval_temperature = eval_temperature
        
        # Setup mixed precision policy
        if use_mixed_precision:
            self.mp_policy = jmp.get_policy('half')
        else:
            self.mp_policy = jmp.get_policy('full')
        
        # Initialize model components
        key_emb, key_model, key_sched = jax.random.split(key, 3)
        
        # Initialize embedding layer
        self.embedding = Embedding(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            key=key_emb
        )
        
        # Initialize transformer model  
        self.model = SequenceDiT(
            max_length=max_length,
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            dreams_embedding_size=dreams_emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            key=key_model,
            mp_policy=self.mp_policy
        )
        
        # Initialize diffusion scheduler
        self.scheduler = DiffusionScheduler(
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end
        )
        
        # Initialize generator and evaluator for evaluation
        self.generator = None
        self.evaluator = None
        
        if eval_generation:
            self.evaluator = MolecularEvaluator()
            
            if tokenizer is not None:
                try:
                    self.generator = MolecularGenerator(
                        model=self.model,
                        scheduler=self.scheduler,
                        tokenizer=tokenizer,
                        max_length=max_length,
                        guidance_scale=1.0
                    )
                    print("Generator initialized for evaluation.")
                except Exception as e:
                    print(f"Warning: Could not initialize generator: {e}")
                    self.eval_generation = False
            else:
                print("Warning: tokenizer not provided, generator will be initialized later in train()")
        else:
            print("Evaluation generation disabled.")
        
        # Initialize optimizer
        self.optimizer = optax.adamw(
            learning_rate=learning_rate,
            weight_decay=1e-6,
            b1=0.9,
            b2=0.99
        )
        
        # Initialize optimizer state
        params = eqx.filter(self.model, eqx.is_array)
        self.opt_state = self.optimizer.init(params)
        
        # Training state
        self.step = 0
        self.epoch = 0
        
    def train_epoch(self, data_loader, epoch, key):
        """Train for one epoch."""
        epoch_loss = 0.0
        num_batches = 0
        
        # Create progress bar with step limit info
        desc = f"Epoch {epoch}"
        if self.max_steps_per_epoch is not None:
            desc += f" (max {self.max_steps_per_epoch} steps)"
        pbar = tqdm(data_loader, desc=desc)
        
        for batch_idx, batch in enumerate(pbar):
            # Check if we've reached the maximum steps per epoch
            if self.max_steps_per_epoch is not None and num_batches >= self.max_steps_per_epoch:
                print(f"Reached max steps per epoch ({self.max_steps_per_epoch}), stopping early")
                break
                
            # Generate random key for this batch
            step_key = jax.random.fold_in(key, self.step)
            
            # Keep only JAX array entries that are consumed inside the JIT to avoid retracing
            train_batch = {
                'tokens': batch['tokens'],
                'dreams_embedding': batch['dreams_embedding'],
            }

            # Perform training step
            self.model, self.opt_state, loss = train_step(
                self.model, self.opt_state, train_batch, step_key, self.optimizer, self.scheduler
            )
            
            # Update metrics
            epoch_loss += float(loss)
            num_batches += 1
            self.step += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss:.4f}', 'avg_loss': f'{epoch_loss/num_batches:.4f}'})
            
            # Log to wandb if available
            if wandb.run is not None:
                wandb.log({
                    'train_loss': float(loss),
                    'step': self.step,
                    'epoch': epoch
                })
        
        return epoch_loss / num_batches if num_batches > 0 else 0.0
    
    def validate_epoch(self, data_loader, epoch, key):
        """Validate for one epoch."""
        epoch_loss = 0.0
        num_batches = 0
        
        # Create progress bar with step limit info
        desc = f"Validation {epoch}"
        if self.max_steps_per_epoch is not None:
            desc += f" (max {self.max_steps_per_epoch} steps)"
        pbar = tqdm(data_loader, desc=desc)
        
        for batch_idx, batch in enumerate(pbar):
            # Check if we've reached the maximum steps per epoch
            if self.max_steps_per_epoch is not None and num_batches >= self.max_steps_per_epoch:
                print(f"Reached max validation steps per epoch ({self.max_steps_per_epoch}), stopping early")
                break
                
            # Generate random key for this batch
            step_key = jax.random.fold_in(key, batch_idx)
            
            # Only keep relevant array fields to avoid retracing inside JITed loss_fn
            val_batch = {
                'tokens': batch['tokens'],
                'dreams_embedding': batch['dreams_embedding'],
            }

            # Compute validation loss
            loss = loss_fn(self.model, val_batch, step_key, self.scheduler)
            
            # Update metrics
            epoch_loss += float(loss)
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'val_loss': f'{loss:.4f}', 'avg_val_loss': f'{epoch_loss/num_batches:.4f}'})
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        
        # Generation evaluation
        gen_metrics = {}
        if self.eval_generation and self.generator is not None and self.evaluator is not None:
            print(f"\nRunning generation evaluation on {self.eval_max_batches} batches...")
            eval_key = jax.random.fold_in(key, 9999)  # Use a different key for evaluation
            
            # Create evaluation batches with ground truth SMILES
            eval_batches = []
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx >= self.eval_max_batches:
                    break
                eval_batch = {
                    'dreams_embedding': batch['dreams_embedding'],
                    'smiles': batch.get('smiles', [])  # Ground truth SMILES if available
                }
                eval_batches.append(eval_batch)
            
            # Run generation evaluation
            if eval_batches:
                total_generated = 0
                total_valid = 0
                total_similarity = 0.0
                similarity_count = 0
                all_smiles_pairs = []  # Collect all generated/reference pairs
                
                for batch_idx, eval_batch in enumerate(eval_batches):
                    batch_key = jax.random.fold_in(eval_key, batch_idx)
                    batch_metrics = self.evaluator.evaluate_batch(
                        self.generator,
                        batch_key,
                        eval_batch,
                        temperature=self.eval_temperature
                    )
                    
                    total_generated += batch_metrics['num_generated']
                    total_valid += batch_metrics['num_valid']
                    if batch_metrics['num_valid'] > 0:
                        total_similarity += batch_metrics['avg_similarity'] * batch_metrics['num_valid']
                        similarity_count += batch_metrics['num_valid']
                    
                    # Collect SMILES pairs for logging (use already computed metrics)
                    generated_smiles = batch_metrics.get('generated_smiles', [])
                    gt_smiles = batch_metrics.get('gt_smiles', [])
                    
                    for i, (gen_smiles, ref_smiles) in enumerate(zip(generated_smiles, gt_smiles)):
                        all_smiles_pairs.append({
                            'batch_idx': batch_idx,
                            'sample_idx': i,
                            'generated_smiles': gen_smiles,
                            'reference_smiles': ref_smiles or "N/A"
                        })
                
                # Calculate overall metrics
                gen_metrics = {
                    'gen_validity_rate': total_valid / total_generated if total_generated > 0 else 0.0,
                    'gen_avg_similarity': total_similarity / similarity_count if similarity_count > 0 else 0.0,
                    'gen_num_generated': total_generated,
                    'gen_num_valid': total_valid
                }
                
                # Log SMILES pairs to wandb
                if wandb.run is not None and all_smiles_pairs:
                    # Create a wandb table for SMILES pairs
                    smiles_table = wandb.Table(
                        columns=["Batch", "Sample", "Generated SMILES", "Reference SMILES"]
                    )
                    
                    # Add rows to the table (limit to prevent overwhelming logs)
                    max_rows_to_log = 50  # Limit number of rows for readability
                    for pair in all_smiles_pairs[:max_rows_to_log]:
                        smiles_table.add_data(
                            pair['batch_idx'],
                            pair['sample_idx'],
                            pair['generated_smiles'],
                            pair['reference_smiles']
                        )
                    
                    gen_metrics['smiles_pairs'] = smiles_table
                
                print(f"Generation metrics: Validity={gen_metrics['gen_validity_rate']:.3f}, "
                      f"Similarity={gen_metrics['gen_avg_similarity']:.3f}, "
                      f"Valid={gen_metrics['gen_num_valid']}/{gen_metrics['gen_num_generated']}")
                
                # Print a few examples
                print(f"\nExample generated/reference pairs:")
                for i, pair in enumerate(all_smiles_pairs[:5]):  # Show first 5 examples
                    print(f"  {i+1}. Generated: {pair['generated_smiles'][:50]}{'...' if len(pair['generated_smiles']) > 50 else ''}")
                    print(f"     Reference: {pair['reference_smiles'][:50]}{'...' if len(pair['reference_smiles']) > 50 else ''}")
                if len(all_smiles_pairs) > 5:
                    print(f"  ... and {len(all_smiles_pairs) - 5} more pairs")
        
        # Log to wandb if available
        if wandb.run is not None:
            log_dict = {
                'val_loss': avg_loss,
                'epoch': epoch
            }
            log_dict.update(gen_metrics)  # Add generation metrics
            wandb.log(log_dict)
        
        return avg_loss
    
    def save_checkpoint(self, checkpoint_dir: str, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:03d}_val_{val_loss:.4f}.pkl')
        
        checkpoint = {
            'model': self.model,
            'embedding': self.embedding,
            'opt_state': self.opt_state,
            'scheduler': self.scheduler,
            'epoch': epoch,
            'step': self.step,
            'val_loss': val_loss,
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.model = checkpoint['model']
        self.embedding = checkpoint['embedding'] 
        self.opt_state = checkpoint['opt_state']
        self.scheduler = checkpoint['scheduler']
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        
        print(f"Loaded checkpoint: {checkpoint_path}")
    
    def train(
        self,
        data_module: MolecularDataModule,
        max_epochs: int,
        checkpoint_dir: str,
        save_every: int = 5,
        key: Optional[jax.random.PRNGKey] = None
    ):
        """Main training loop."""
        if key is None:
            key = jax.random.PRNGKey(int(time.time()))
        
        # Get data loaders (data_module.setup("fit") should already be called)
        train_loader = data_module.get_train_dataloader()
        val_loader = data_module.get_val_dataloader()
        
        # Initialize generator for evaluation if not already done and evaluation is enabled
        if self.eval_generation and self.generator is None and self.evaluator is not None:
            try:
                tokenizer = data_module.tokenizer
                self.generator = MolecularGenerator(
                    model=self.model,
                    scheduler=self.scheduler,
                    tokenizer=tokenizer,
                    max_length=self.max_length,
                    guidance_scale=1.0
                )
                print("Generator initialized for evaluation (fallback).")
            except Exception as e:
                print(f"Warning: Could not initialize generator: {e}")
                self.eval_generation = False
        
        print(f"Starting training for {max_epochs} epochs...")
        print(f"Training samples: {len(data_module.train_dataset)}")
        print(f"Validation samples: {len(data_module.val_dataset)}")
        
        # Debug info about evaluation setup
        if self.eval_generation:
            if self.generator is not None:
                print(f"✓ Generation evaluation enabled with {self.eval_max_batches} max batches")
            else:
                print("✗ Generation evaluation enabled but generator not initialized")
        else:
            print("✗ Generation evaluation disabled")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.epoch, max_epochs):
            epoch_key = jax.random.fold_in(key, epoch)
            train_key, val_key = jax.random.split(epoch_key)
            
            # Training phase
            print(f"\n=== Epoch {epoch + 1}/{max_epochs} ===")
            train_loss = self.train_epoch(train_loader, epoch + 1, train_key)
            
            # Validation phase
            val_loss = self.validate_epoch(val_loader, epoch + 1, val_key)
            
            self.epoch = epoch + 1
            
            print(f"Epoch {self.epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Save checkpoint if validation improved or every save_every epochs
            if val_loss < best_val_loss or (epoch + 1) % save_every == 0:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"New best validation loss: {val_loss:.4f}")
                
                # self.save_checkpoint(checkpoint_dir, self.epoch, val_loss)


def main():
    parser = argparse.ArgumentParser(description="Train molecular diffusion model with JAX")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--vocab_file", type=str, default="data/vocab.txt", help="Vocabulary file")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers")
    
    # Model parameters
    parser.add_argument("--embedding_size", type=int, default=384, help="Model embedding dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--mlp_ratio", type=float, default=4.0, help="MLP expansion ratio")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum epochs")
    parser.add_argument("--max_steps_per_epoch", type=int, default=None, help="Maximum steps per epoch (None means use all data)")
    parser.add_argument("--num_timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--beta_start", type=float, default=1e-4, help="Diffusion beta start")
    parser.add_argument("--beta_end", type=float, default=4e-2, help="Diffusion beta end")
    
    # Training setup
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--use_mixed_precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Generation evaluation parameters
    parser.add_argument("--eval_generation", action="store_true", help="Enable generation evaluation during validation")
    parser.add_argument("--eval_max_batches", type=int, default=2, help="Maximum batches for generation evaluation")
    parser.add_argument("--eval_temperature", type=float, default=1.0, help="Temperature for generation sampling")
    
    # Wandb logging
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="molecular-diffusion-jax", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity/team name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None, help="Wandb tags for the run")
    
    # Resume training
    parser.add_argument("--resume_from", type=str, default=None, help="Resume training from checkpoint")
    
    args = parser.parse_args()
    
    # Set random seed
    key = jax.random.PRNGKey(args.seed)
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            tags=args.wandb_tags,
            config=vars(args)
        )
        print(f"Using Weights & Biases logging - Project: {args.wandb_project}")
    
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
    
    # Setup data module to initialize tokenizer
    tokenizer = SmilesTokenizer(vocab_file=args.vocab_file)
    
    # Calculate hidden size from mlp_ratio
    hidden_size = int(args.embedding_size * args.mlp_ratio)
    
    # Initialize trainer
    trainer = MolecularDiffusionTrainer(
        vocab_size=dataset_info['vocab_size'],
        embedding_size=args.embedding_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        hidden_size=hidden_size,
        dreams_emb_size=dataset_info['dreams_emb_dim'],
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        num_timesteps=args.num_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        use_mixed_precision=args.use_mixed_precision,
        eval_generation=args.eval_generation,
        eval_max_batches=args.eval_max_batches,
        eval_temperature=args.eval_temperature,
        max_steps_per_epoch=args.max_steps_per_epoch,
        tokenizer=tokenizer,
        key=key
    )
    
    # Resume from checkpoint if specified
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
    
    # Print model info
    print(f"\nDataset Summary:")
    print(f"Dreams embedding dimension: {dataset_info['dreams_emb_dim']}")
    print(f"Vocabulary size: {dataset_info['vocab_size']}")
    print(f"Number of instruments: {dataset_info['num_instruments']}")
    print(f"Number of adducts: {dataset_info['num_adducts']}")
    
    # Count model parameters
    model_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(trainer.model, eqx.is_array)))
    embedding_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(trainer.embedding, eqx.is_array)))
    total_params = model_params + embedding_params
    
    print(f"\nModel Summary:")
    print(f"Model parameters: {model_params:,}")
    print(f"Embedding parameters: {embedding_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Mixed precision: {args.use_mixed_precision}")
    
    # Start training
    trainer.train(
        data_module=data_module,
        max_epochs=args.max_epochs,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        key=key
    )
    
    print(f"\nTraining completed! Checkpoints saved to: {args.checkpoint_dir}")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
