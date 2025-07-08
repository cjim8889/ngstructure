from typing import Dict, Optional

import jax

from pkg.generator import MolecularGenerator

# RDKit imports for molecule validation and similarity
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Molecular evaluation will be disabled.")


class MolecularEvaluator:
    """Evaluates molecular generation quality using validity and similarity metrics."""
    
    def __init__(self):
        """Initialize the evaluator."""
        if not RDKIT_AVAILABLE:
            print("Warning: RDKit not available. Evaluation will return zero metrics.")
    
    def _calculate_tanimoto_similarity(self, mol1, mol2) -> float:
        """Calculate Tanimoto similarity between two RDKit molecules."""
        if not RDKIT_AVAILABLE or mol1 is None or mol2 is None:
            return 0.0
        
        try:
            fpgen = AllChem.GetRDKitFPGenerator()
            fp1 = fpgen.GetFingerprint(mol1)
            fp2 = fpgen.GetFingerprint(mol2)
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        except:
            return 0.0
    
    def _is_valid_smiles(self, smiles: str) -> bool:
        """Check if a SMILES string represents a valid molecule."""
        if not RDKIT_AVAILABLE:
            return False
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def _smiles_to_mol(self, smiles: str):
        """Convert SMILES string to RDKit molecule object."""
        if not RDKIT_AVAILABLE:
            return None
        
        try:
            return Chem.MolFromSmiles(smiles)
        except:
            return None
    
    def evaluate_batch(
        self,
        generator: MolecularGenerator,
        key: jax.random.PRNGKey,
        val_batch: Dict,
        *,
        temperature: float = 1.0,
        num_sampling_steps: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate a validation batch by generating molecules and computing metrics.
        
        Args:
            generator: MolecularGenerator instance
            key: Random key for generation
            val_batch: Validation batch containing 'dreams_embedding' and 'smiles'
            temperature: Sampling temperature
            num_sampling_steps: Number of diffusion sampling steps
            
        Returns:
            Dictionary with evaluation metrics:
            - validity_rate: Fraction of generated SMILES that are valid molecules
            - avg_similarity: Average Tanimoto similarity with ground truth
            - num_generated: Total number of molecules generated
            - num_valid: Number of valid molecules generated
        """
        if not RDKIT_AVAILABLE:
            return {
                'validity_rate': 0.0,
                'avg_similarity': 0.0,
                'num_generated': 0,
                'num_valid': 0
            }
        
        dreams_embeddings = val_batch['dreams_embedding']  # [batch_size, dreams_dim]
        gt_smiles = val_batch.get('smiles', [])  # Ground truth SMILES
        
        batch_size = dreams_embeddings.shape[0]
        valid_count = 0
        total_similarity = 0.0
        similarity_count = 0

        sample_key = jax.random.split(key, batch_size)
        dreams_emb = dreams_embeddings  # [dreams_dim]
        
        # Generate tokens
        tokens = jax.vmap(lambda key, emb: generator.sample(key, emb, temperature=temperature, num_sampling_steps=num_sampling_steps))(sample_key, dreams_emb)
        
        generated_smiles_list = []
        # Generate molecules for each sample in the batch
        for i in range(batch_size):    
            # Decode to SMILES
            generated_smiles = generator.decode(tokens[i])
            generated_smiles_list.append(generated_smiles)
            # Check validity with RDKit
            if self._is_valid_smiles(generated_smiles):
                valid_count += 1
                generated_mol = self._smiles_to_mol(generated_smiles)
                
                # Calculate similarity with ground truth if available
                if i < len(gt_smiles) and gt_smiles[i]:
                    gt_mol = self._smiles_to_mol(gt_smiles[i])
                    if gt_mol is not None and generated_mol is not None:
                        similarity = self._calculate_tanimoto_similarity(generated_mol, gt_mol)
                        total_similarity += similarity
                        similarity_count += 1
        
        # Calculate metrics
        validity_rate = valid_count / batch_size if batch_size > 0 else 0.0
        avg_similarity = total_similarity / similarity_count if similarity_count > 0 else 0.0
        
        return {
            'validity_rate': validity_rate,
            'avg_similarity': avg_similarity,
            'num_generated': batch_size,
            'num_valid': valid_count,
            'generated_smiles': generated_smiles_list,
            'gt_smiles': gt_smiles
        }
    
    def evaluate_epoch(
        self,
        generator,
        key: jax.random.PRNGKey,
        val_dataloader,
        *,
        max_batches: Optional[int] = None,
        temperature: float = 1.0,
        num_sampling_steps: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate the generator on validation data for one epoch.
        
        Args:
            generator: MolecularGenerator instance
            key: Random key for generation
            val_dataloader: Validation data loader
            max_batches: Maximum number of batches to evaluate (None for all)
            temperature: Sampling temperature
            num_sampling_steps: Number of diffusion sampling steps
            
        Returns:
            Dictionary with aggregated evaluation metrics
        """
        if not RDKIT_AVAILABLE:
            print("Warning: RDKit not available. Skipping molecular evaluation.")
            return {
                'validity_rate': 0.0,
                'avg_similarity': 0.0,
                'num_generated': 0,
                'num_valid': 0
            }
        
        total_generated = 0
        total_valid = 0
        total_similarity = 0.0
        similarity_count = 0
        
        for batch_idx, batch in enumerate(val_dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            batch_key = jax.random.fold_in(key, batch_idx)
            
            # Evaluate this batch
            batch_metrics = self.evaluate_batch(
                generator,
                batch_key,
                batch,
                temperature=temperature,
                num_sampling_steps=num_sampling_steps
            )
            
            # Accumulate metrics
            total_generated += batch_metrics['num_generated']
            total_valid += batch_metrics['num_valid']
            
            # Weight similarity by number of valid comparisons in this batch
            if batch_metrics['num_valid'] > 0:
                total_similarity += batch_metrics['avg_similarity'] * batch_metrics['num_valid']
                similarity_count += batch_metrics['num_valid']
        
        # Calculate final metrics
        overall_validity = total_valid / total_generated if total_generated > 0 else 0.0
        overall_similarity = total_similarity / similarity_count if similarity_count > 0 else 0.0
        
        return {
            'validity_rate': overall_validity,
            'avg_similarity': overall_similarity,
            'num_generated': total_generated,
            'num_valid': total_valid
        } 