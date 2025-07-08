import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
import numpy as np
from deepchem.feat.smiles_tokenizer import SmilesTokenizer


class MolecularGenerator:
    """Generator for sampling molecules from trained diffusion model."""
    
    def __init__(
        self,
        model,
        scheduler,
        tokenizer: SmilesTokenizer,
        max_length: int = 500,
        guidance_scale: float = 1.0
    ):
        """
        Args:
            model: Trained DiT model
            scheduler: DiffusionScheduler used for training
            tokenizer: SMILES tokenizer for decoding
            max_length: Maximum sequence length
            guidance_scale: Classifier-free guidance scale
        """
        self.model = model
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.guidance_scale = guidance_scale
        
        # Pre-compute token embeddings for nearest neighbor search
        self._build_token_embedding_cache()
    
    def _build_token_embedding_cache(self):
        """Pre-compute embeddings for all vocabulary tokens for fast NN search."""
        device = next(self.model.parameters()).device
        vocab_size = self.tokenizer.vocab_size
        
        # Get embeddings for all tokens
        all_tokens = torch.arange(vocab_size, device=device)
        with torch.no_grad():
            self.token_embeddings = self.model.embed_x(all_tokens)  # [vocab_size, d_model]
            # Normalize embeddings for cosine similarity
            self.token_embeddings = F.normalize(self.token_embeddings, dim=-1)
    
    def embedding_to_tokens(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Convert embeddings to tokens using nearest neighbor search.
        
        Args:
            embeddings: [batch_size, seq_len, d_model]
            
        Returns:
            tokens: [batch_size, seq_len]
        """
        batch_size, seq_len, d_model = embeddings.shape
        
        # Normalize embeddings for cosine similarity
        embeddings_norm = F.normalize(embeddings, dim=-1)  # [B, L, D]
        
        # Compute cosine similarity with all token embeddings
        # embeddings_norm: [B, L, D], token_embeddings: [V, D]
        similarities = torch.matmul(
            embeddings_norm.view(batch_size * seq_len, d_model),  # [B*L, D]
            self.token_embeddings.t()  # [D, V]
        )  # [B*L, V]
        
        # Find nearest tokens
        nearest_tokens = torch.argmax(similarities, dim=-1)  # [B*L]
        
        return nearest_tokens.view(batch_size, seq_len)
    
    @torch.no_grad()
    def sample(
        self,
        dreams_embeddings: torch.Tensor,
        num_samples: int = 1,
        temperature: float = 1.0,
        num_sampling_steps: Optional[int] = None
    ) -> Tuple[torch.Tensor, List[str]]:
        """Sample sequences using reverse diffusion.
        
        Args:
            dreams_embeddings: [batch_size, dreams_emb_dim] conditioning
            num_samples: Number of samples to generate
            temperature: Sampling temperature
            num_sampling_steps: Number of denoising steps (default: use all timesteps)
            
        Returns:
            tokens: [batch_size, max_length] generated token sequences
            smiles_list: List of decoded SMILES strings
        """
        device = dreams_embeddings.device
        batch_size = dreams_embeddings.shape[0]
        
        if num_sampling_steps is None:
            num_sampling_steps = self.scheduler.num_timesteps
        
        # Create sampling schedule
        timesteps = torch.linspace(
            self.scheduler.num_timesteps - 1, 0, 
            num_sampling_steps, 
            dtype=torch.long, 
            device=device
        )
        
        # Start from pure noise
        x_t = torch.randn(
            batch_size, self.max_length, self.model.d_model,
            device=device
        ) * temperature
        
        # Reverse diffusion process
        for i, t in enumerate(timesteps):
            # Prepare time embedding
            t_batch = t.repeat(batch_size)
            
            # Predict the score
            predicted_score = self.model(x_t, t_batch, dreams_embeddings)
            
            # Compute x_{t-1} using the predicted score
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                x_t = self._reverse_step(x_t, predicted_score, t, t_next)
            else:
                # Last step: predict x_0 directly
                x_t = self._predict_x0(x_t, predicted_score, t)
        
        # Convert embeddings to tokens
        tokens = self.embedding_to_tokens(x_t)
        
        # Decode to SMILES
        smiles_list = self._decode_tokens_to_smiles(tokens)
        
        return tokens, smiles_list
    
    def _reverse_step(
        self, 
        x_t: torch.Tensor, 
        predicted_score: torch.Tensor, 
        t: torch.Tensor, 
        t_next: torch.Tensor
    ) -> torch.Tensor:
        """Single reverse diffusion step."""
        device = x_t.device
        
        # Get scheduler coefficients
        sqrt_alphas_cumprod = self.scheduler.sqrt_alphas_cumprod.to(device)
        sqrt_one_minus_alphas_cumprod = self.scheduler.sqrt_one_minus_alphas_cumprod.to(device)
        
        # Predict x_0 from score
        sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        # Predict x_0 using score function
        pred_x0 = (x_t + sqrt_one_minus_alpha_cumprod_t * predicted_score) / sqrt_alpha_cumprod_t
        
        # Compute x_{t-1}
        sqrt_alpha_cumprod_t_next = sqrt_alphas_cumprod[t_next].view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t_next = sqrt_one_minus_alphas_cumprod[t_next].view(-1, 1, 1)
        
        x_t_next = sqrt_alpha_cumprod_t_next * pred_x0 + sqrt_one_minus_alpha_cumprod_t_next * predicted_score
        
        # Add noise for stochastic sampling (except for the last step)
        if t_next > 0:
            noise = torch.randn_like(x_t)
            sigma = sqrt_one_minus_alpha_cumprod_t_next * 0.1  # Small noise term
            x_t_next = x_t_next + sigma * noise
        
        return x_t_next
    
    def _predict_x0(
        self, 
        x_t: torch.Tensor, 
        predicted_score: torch.Tensor, 
        t: torch.Tensor
    ) -> torch.Tensor:
        """Predict clean x_0 from noisy x_t and predicted score."""
        device = x_t.device
        sqrt_alphas_cumprod = self.scheduler.sqrt_alphas_cumprod.to(device)
        sqrt_one_minus_alphas_cumprod = self.scheduler.sqrt_one_minus_alphas_cumprod.to(device)
        
        sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        # Predict x_0 using score function
        pred_x0 = (x_t + sqrt_one_minus_alpha_cumprod_t * predicted_score) / sqrt_alpha_cumprod_t
        
        return pred_x0
    
    def _decode_tokens_to_smiles(self, tokens: torch.Tensor) -> List[str]:
        """Decode token sequences to SMILES strings."""
        smiles_list = []
        
        for token_seq in tokens:
            # Convert to numpy and remove padding
            token_seq_np = token_seq.cpu().numpy()
            
            # Remove padding tokens
            valid_tokens = []
            for token in token_seq_np:
                if token == self.tokenizer.pad_token_id:
                    break
                valid_tokens.append(int(token))
            
            # Decode to SMILES
            try:
                smiles = self.tokenizer.decode(valid_tokens)
                smiles_list.append(smiles)
            except Exception as e:
                # Handle decoding errors gracefully
                smiles_list.append(f"<DECODE_ERROR>")
        
        return smiles_list
    
    def evaluate_sample_quality(
        self, 
        generated_smiles: List[str], 
        reference_smiles: Optional[List[str]] = None
    ) -> dict:
        """Evaluate the quality of generated SMILES using RDKit."""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, Crippen
            from rdkit import RDLogger
            # Suppress RDKit warnings
            RDLogger.DisableLog('rdApp.*')
        except ImportError:
            print("Warning: RDKit not available, falling back to basic validation")
            return self._basic_evaluate_sample_quality(generated_smiles, reference_smiles)
        
        metrics = {}
        valid_mols = []
        valid_smiles = []
        
        # Validity check using RDKit
        for smiles in generated_smiles:
            if smiles.startswith("<") or len(smiles.strip()) == 0:
                continue
                
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Additional check: molecule should have at least one atom
                    if mol.GetNumAtoms() > 0:
                        valid_mols.append(mol)
                        valid_smiles.append(smiles)
            except:
                continue
        
        total_generated = len(generated_smiles)
        valid_count = len(valid_mols)
        
        
        metrics['validity'] = valid_count / total_generated if total_generated > 0 else 0.0
        metrics['total_generated'] = total_generated
        metrics['valid_generated'] = valid_count
        
        if valid_smiles:
            # Uniqueness (on valid SMILES)
            # Canonicalize SMILES to properly check uniqueness
            canonical_smiles = []
            for mol in valid_mols:
                try:
                    canonical = Chem.MolToSmiles(mol, canonical=True)
                    canonical_smiles.append(canonical)
                except:
                    continue
            
            unique_canonical = set(canonical_smiles)
            metrics['uniqueness'] = len(unique_canonical) / len(canonical_smiles) if canonical_smiles else 0.0
            metrics['unique_valid'] = len(unique_canonical)
            
            # Molecular properties
            try:
                # Molecular weight
                mol_weights = [Descriptors.MolWt(mol) for mol in valid_mols]
                metrics['avg_mol_weight'] = np.mean(mol_weights)
                metrics['std_mol_weight'] = np.std(mol_weights)
                
                # LogP (lipophilicity)
                logp_values = [Crippen.MolLogP(mol) for mol in valid_mols]
                metrics['avg_logp'] = np.mean(logp_values)
                metrics['std_logp'] = np.std(logp_values)
                
                # Number of atoms
                num_atoms = [mol.GetNumAtoms() for mol in valid_mols]
                metrics['avg_num_atoms'] = np.mean(num_atoms)
                metrics['std_num_atoms'] = np.std(num_atoms)
                
                # Number of bonds
                num_bonds = [mol.GetNumBonds() for mol in valid_mols]
                metrics['avg_num_bonds'] = np.mean(num_bonds)
                
                # Number of rings
                num_rings = [Descriptors.RingCount(mol) for mol in valid_mols]
                metrics['avg_num_rings'] = np.mean(num_rings)
                
            except Exception as e:
                print(f"Warning: Could not compute molecular properties: {e}")
        else:
            metrics['uniqueness'] = 0.0
            metrics['unique_valid'] = 0
            metrics['avg_mol_weight'] = 0.0
            metrics['avg_logp'] = 0.0
            metrics['avg_num_atoms'] = 0.0
            metrics['avg_num_bonds'] = 0.0
            metrics['avg_num_rings'] = 0.0
        
        # String-based metrics
        lengths = [len(s) for s in generated_smiles if not s.startswith("<")]
        metrics['avg_smiles_length'] = np.mean(lengths) if lengths else 0.0
        
        # Diversity metrics (if reference provided)
        if reference_smiles and valid_smiles:
            try:
                # Convert reference SMILES to molecules
                ref_mols = []
                for ref_smiles in reference_smiles:
                    mol = Chem.MolFromSmiles(ref_smiles)
                    if mol is not None:
                        ref_mols.append(mol)
                
                if ref_mols:
                    # Calculate Tanimoto similarity using Morgan fingerprints
                    from rdkit.Chem import rdMolDescriptors
                    from rdkit import DataStructs
                    
                    # Get fingerprints for generated molecules
                    gen_fps = []
                    for mol in valid_mols:
                        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                        gen_fps.append(fp)
                    
                    # Get fingerprints for reference molecules
                    ref_fps = []
                    for mol in ref_mols:
                        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                        ref_fps.append(fp)
                    
                    # Calculate similarities
                    similarities = []
                    for gen_fp in gen_fps:
                        max_sim = 0.0
                        for ref_fp in ref_fps:
                            sim = DataStructs.TanimotoSimilarity(gen_fp, ref_fp)
                            max_sim = max(max_sim, sim)
                        similarities.append(max_sim)
                    
                    metrics['avg_tanimoto_similarity'] = np.mean(similarities) if similarities else 0.0
                    metrics['max_tanimoto_similarity'] = np.max(similarities) if similarities else 0.0
                    
            except Exception as e:
                print(f"Warning: Could not compute diversity metrics: {e}")
        
        return metrics
    
    def _basic_evaluate_sample_quality(
        self, 
        generated_smiles: List[str], 
        reference_smiles: Optional[List[str]] = None
    ) -> dict:
        """Fallback evaluation without RDKit."""
        metrics = {}
        
        # Basic validity check
        valid_count = 0
        for smiles in generated_smiles:
            if not smiles.startswith("<") and len(smiles) > 0:
                try:
                    # Simple SMILES validation - check for basic SMILES characters
                    if any(c in smiles for c in ['C', 'N', 'O', 'S', 'P']):
                        valid_count += 1
                except:
                    pass
        
        metrics['validity'] = valid_count / len(generated_smiles) if generated_smiles else 0.0
        metrics['total_generated'] = len(generated_smiles)
        metrics['valid_generated'] = valid_count
        
        # Uniqueness
        unique_smiles = set(generated_smiles)
        metrics['uniqueness'] = len(unique_smiles) / len(generated_smiles) if generated_smiles else 0.0
        
        # Average length
        lengths = [len(s) for s in generated_smiles if not s.startswith("<")]
        metrics['avg_smiles_length'] = np.mean(lengths) if lengths else 0.0
        
        return metrics


class EpochEvaluator:
    """Handles evaluation after each training epoch."""
    
    def __init__(
        self,
        generator: MolecularGenerator,
        num_samples: int = 16,
        log_interval: int = 1
    ):
        """
        Args:
            generator: MolecularGenerator instance
            num_samples: Number of samples to generate for evaluation
            log_interval: Evaluate every N epochs
        """
        self.generator = generator
        self.num_samples = num_samples
        self.log_interval = log_interval
        self.evaluation_history = []
    
    def evaluate(self, pl_module, epoch: int) -> dict:
        """Run evaluation and return metrics."""
        if epoch % self.log_interval != 0:
            return {}
        
        # Get a batch of dreams embeddings from validation set
        val_dataloader = pl_module.trainer.datamodule.val_dataloader()
        val_batch = next(iter(val_dataloader))
        
        # Select subset for generation
        dreams_emb = val_batch['dreams_embedding'][:self.num_samples].to(pl_module.device)
        reference_smiles = val_batch['smiles'][:self.num_samples]
        
        # Generate samples
        self.generator.model.eval()
        generated_tokens, generated_smiles = self.generator.sample(
            dreams_embeddings=dreams_emb,
            num_samples=self.num_samples
        )
        
        # Evaluate quality
        metrics = self.generator.evaluate_sample_quality(
            generated_smiles, reference_smiles
        )
        
        # Log metrics
        for key, value in metrics.items():
            pl_module.log(f'eval/{key}', value, on_epoch=True, prog_bar=True)
        
        # Store some examples for logging
        examples = {
            'generated_smiles': generated_smiles[:5],  # First 5 examples
            'reference_smiles': reference_smiles[:5],
            'epoch': epoch
        }
        
        self.evaluation_history.append({
            'epoch': epoch,
            'metrics': metrics,
            'examples': examples
        })
        
        return metrics 