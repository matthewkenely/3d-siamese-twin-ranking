import numpy as np
import scipy
import random
import os
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms
from torchvision import models

from torch.utils.data import Dataset, DataLoader, random_split



def triplet_loss(anchor, positive, negative, 
                             margin=1.0, distance_metric='cosine'):
    """
    Triplet loss with masking for sparse MRI data.
    
    Args:
        anchor, positive, negative: Feature embeddings [batch_size, embedding_dim] or [batch_size, channels, d, h, w]
        anchor_mask, positive_mask, negative_mask: Binary masks indicating valid regions
    """
    
    pos_distance = 1 - F.cosine_similarity(anchor, positive)
    neg_distance = 1 - F.cosine_similarity(anchor, negative)

    # Triplet loss with margin
    loss = F.relu(pos_distance - neg_distance + margin)
    
    # Statistics for active triplets
    active_triplets = (loss > 0).float()
    num_active = active_triplets.sum()
    
    # if num_active > 0:
    #     loss_mean = loss.sum() / num_active
    #     pos_mean = (pos_distance * active_triplets).sum() / num_active
    #     neg_mean = (neg_distance * active_triplets).sum() / num_active
    # else:
    #     loss_mean = torch.tensor(0.0, device=loss.device)
    #     pos_mean = pos_distance.mean()
    #     neg_mean = neg_distance.mean()

    loss_mean = loss.mean()
    pos_mean = pos_distance.mean()
    neg_mean = neg_distance.mean()
    
    return loss_mean, pos_mean, neg_mean, num_active / len(loss)


class TripletBrainDataset(Dataset):
    def __init__(self, data_path, split='train', transform=None, 
                 mining_strategy='random', num_negatives_per_positive=5,
                 batch_embedding_size=32, precompute_embeddings=True, margin=0.5):
        """
        Optimized triplet dataset for MRI twin recognition.
        Now uses centralized subjects directory instead of split-specific copies.
        
        Args:
            data_path: Base path to dataset (e.g., './t1/')
            split: 'train', 'val', or 'test'
            transform: Optional transforms
            mining_strategy: 'random', 'hard', or 'semi_hard'
            num_negatives_per_positive: How many negative samples per positive pair
            batch_embedding_size: Batch size for embedding computation
            precompute_embeddings: Whether to precompute all embeddings at start
        """
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self.mining_strategy = mining_strategy
        self.num_negatives_per_positive = num_negatives_per_positive
        self.batch_embedding_size = batch_embedding_size
        self.precompute_embeddings = precompute_embeddings
        self.margin = margin
        
        # Load twin pairs for this split
        self.twin_pairs = []  # List of (subject1_id, subject2_id)
        self.all_subjects = []  # List of all subject IDs
        self.subject_to_twin_idx = {}  # Maps subject ID to twin pair index
        
        self._load_data()
        
        # Path to centralized subject files (no longer split-specific)
        self.subjects_path = self.data_path / 'subjects'
        
        # Optimized embedding storage
        self.cached_embeddings = {}
        self.embedding_matrix = None  # Precomputed embeddings as tensor
        self.subject_to_idx = {}  # Maps subject ID to embedding matrix index
        self.model = None
        self.model_device = None
        
        # Cache for loaded and normalized images
        self.cached_images = {}
        
        # Pre-filter candidates by twin pairs for faster lookup
        self.negative_candidates = {}  # Maps subject_id -> list of valid negatives
        self._precompute_negative_candidates()

        
    def _load_data(self):
        """Load twin pairs and all subjects from metadata files."""
        # Load twin pairs for this split
        pairs_file = self.data_path / self.split / 'twin_pairs.txt'
        with open(pairs_file, 'r') as f:
            for i, line in enumerate(f):
                subject1, subject2 = line.strip().split(',')
                self.twin_pairs.append((subject1, subject2))
                
                # Map subjects to twin pair index
                self.subject_to_twin_idx[subject1] = i
                self.subject_to_twin_idx[subject2] = i
        
        # Load all subjects (for negative sampling) - now from split directory
        subjects_file = self.data_path / self.split / 'all_subjects.txt'
        with open(subjects_file, 'r') as f:
            for line in f:
                self.all_subjects.append(line.strip())
    
    def _load_image(self, subject_id):
        """Load image for a subject ID from centralized subjects directory."""
        # Check cache first
        if subject_id in self.cached_images:
            return self.cached_images[subject_id]
            
        # Load from centralized subjects directory
        subject_file = self.subjects_path / f"{subject_id}.npy"
        img = np.load(subject_file)
        
        # Normalize to [0, 1] range
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:  # Avoid division by zero
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)  # Handle edge case where all values are the same
        
        # Cache the normalized image
        self.cached_images[subject_id] = img
        return img

    def _precompute_negative_candidates(self):
        """Pre-filter negative candidates for each subject."""
        for subject in self.all_subjects:
            twin_idx = self.subject_to_twin_idx.get(subject)
            if twin_idx is not None:
                twin_subject1, twin_subject2 = self.twin_pairs[twin_idx]
                excluded = {twin_subject1, twin_subject2}
            else:
                excluded = {subject}
            
            candidates = [s for s in self.all_subjects if s not in excluded]
            self.negative_candidates[subject] = candidates
    
    def set_model_for_mining(self, model):
        """Set model and optionally precompute all embeddings."""
        self.model = model
        self.model_device = next(model.parameters()).device if model is not None else None
        
        if self.precompute_embeddings and model is not None:
            self._precompute_all_embeddings()
    
    def _precompute_all_embeddings(self):
        """Precompute embeddings for all subjects in batches."""
        
        # Create mapping from subject ID to index
        self.subject_to_idx = {subject: i for i, subject in enumerate(self.all_subjects)}
        
        # Batch process all subjects
        all_embeddings = []
        for i in range(0, len(self.all_subjects), self.batch_embedding_size):
            batch_subjects = self.all_subjects[i:i + self.batch_embedding_size]
            batch_images = []
            
            # Load batch of images
            for subject_id in batch_subjects:
                img = self._load_image(subject_id)
                img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                if self.transform:
                    img_tensor = self.transform(img_tensor)
                batch_images.append(img_tensor)
            
            # Stack into batch tensor
            batch_tensor = torch.cat(batch_images, dim=0).to(self.model_device)
            
            # Compute embeddings for batch
            with torch.no_grad():
                batch_embeddings = self.model(batch_tensor)
                all_embeddings.append(batch_embeddings.cpu())
        
        # Concatenate all embeddings
        self.embedding_matrix = torch.cat(all_embeddings, dim=0)
    
    def _get_embedding(self, subject_id):
        """Get embedding for a subject (optimized version)."""
        if self.embedding_matrix is not None:
            # Use precomputed embeddings
            idx = self.subject_to_idx.get(subject_id)
            if idx is not None:
                return self.embedding_matrix[idx:idx+1]
        
        # Fallback to on-demand computation
        if subject_id in self.cached_embeddings:
            return self.cached_embeddings[subject_id]
            
        if self.model is None:
            return None
            
        # Load and process image
        img = self._load_image(subject_id)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        if self.transform:
            img = self.transform(img)

        img = img.to(self.model_device)
            
        # Get embedding
        with torch.no_grad():
            embedding = self.model(img)
            
        self.cached_embeddings[subject_id] = embedding.cpu()
        return embedding.cpu()
    
    def _select_negative_random(self, anchor_id, positive_id, margin=None):
        """Select negative using random sampling (optimized)."""
        candidates = self.negative_candidates.get(anchor_id, [])
        return random.choice(candidates) if candidates else None
    
    def _select_negative_hard_vectorized(self, anchor_id, positive_id, margin=None):
        """Vectorized hard negative selection."""
        if self.embedding_matrix is None:
            return self._select_negative_hard_original(anchor_id, positive_id)
        
        anchor_idx = self.subject_to_idx.get(anchor_id)
        if anchor_idx is None:
            return self._select_negative_random(anchor_id, positive_id)
        
        candidates = self.negative_candidates.get(anchor_id, [])
        if not candidates:
            return None
            
        candidate_indices = [self.subject_to_idx[c] for c in candidates 
                           if c in self.subject_to_idx]
        
        if not candidate_indices:
            return None
        
        # Vectorized distance computation
        anchor_emb = self.embedding_matrix[anchor_idx:anchor_idx+1]
        candidate_embs = self.embedding_matrix[candidate_indices]
        
        cosine_similarities = torch.mm(anchor_emb, candidate_embs.t())
        distances = 1 - cosine_similarities.squeeze(0)

        hardest_idx = torch.argmin(distances)
        
        return candidates[hardest_idx.item()]
    
    def _select_negative_hard_original(self, anchor_id, positive_id):
        """Original hard negative selection."""
        anchor_emb = self._get_embedding(anchor_id)
        if anchor_emb is None:
            return self._select_negative_random(anchor_id, positive_id)
            
        candidates = self.negative_candidates.get(anchor_id, [])
        if not candidates:
            return None
            
        min_distance = float('inf')
        hardest_negative = None
        
        for candidate in candidates:
            candidate_emb = self._get_embedding(candidate)
            if candidate_emb is not None:
                distance = F.cosine_similarity(anchor_emb, candidate_emb)
                if distance < min_distance:
                    min_distance = distance
                    hardest_negative = candidate
        
        return hardest_negative if hardest_negative else self._select_negative_random(anchor_id, positive_id)
    
    def _select_negative_semi_hard_vectorized(self, anchor_id, positive_id, margin=1.0):
        """Vectorized semi-hard negative selection."""
        if self.embedding_matrix is None:
            return self._select_negative_semi_hard_original(anchor_id, positive_id, margin)
        
        anchor_idx = self.subject_to_idx.get(anchor_id)
        positive_idx = self.subject_to_idx.get(positive_id)
        
        if anchor_idx is None or positive_idx is None:
            return self._select_negative_random(anchor_id, positive_id)
        
        # Get candidate indices
        candidates = self.negative_candidates.get(anchor_id, [])
        if not candidates:
            return None
            
        candidate_indices = [self.subject_to_idx[c] for c in candidates 
                           if c in self.subject_to_idx]
        
        if not candidate_indices:
            return None
        
        # Vectorized distance computation
        anchor_emb = self.embedding_matrix[anchor_idx:anchor_idx+1]  # [1, dim]
        positive_emb = self.embedding_matrix[positive_idx:positive_idx+1]  # [1, dim]
        candidate_embs = self.embedding_matrix[candidate_indices]  # [N, dim]
        
        # Compute distances
        pos_cosine_sim = torch.mm(anchor_emb, positive_emb.t()).squeeze()
        pos_distance = 1 - pos_cosine_sim
        
        neg_cosine_sims = torch.mm(anchor_emb, candidate_embs.t()).squeeze(0)
        neg_distances = 1 - neg_cosine_sims
        
        # Find semi-hard negatives
        semi_hard_mask = (neg_distances > pos_distance) & (neg_distances < pos_distance + margin)
        semi_hard_indices = torch.where(semi_hard_mask)[0]
        
        if len(semi_hard_indices) > 0:
            # Random choice from semi-hard negatives
            chosen_idx = semi_hard_indices[torch.randint(0, len(semi_hard_indices), (1,))]
            return candidates[chosen_idx.item()]
        else:
            # Fallback to hardest negative
            hardest_idx = torch.argmin(neg_distances)
            return candidates[hardest_idx.item()]
    
    def _select_negative_semi_hard_original(self, anchor_id, positive_id, margin=1.0):
        """Original semi-hard selection for fallback."""
        anchor_emb = self._get_embedding(anchor_id)
        positive_emb = self._get_embedding(positive_id)
        
        if anchor_emb is None or positive_emb is None:
            return self._select_negative_random(anchor_id, positive_id)
            
        pos_distance = F.cosine_similarity(anchor_emb, positive_emb)
        
        candidates = self.negative_candidates.get(anchor_id, [])
        if not candidates:
            return None
            
        # Find semi-hard negatives
        semi_hard_candidates = []
        
        for candidate in candidates:
            candidate_emb = self._get_embedding(candidate)
            if candidate_emb is not None:
                neg_distance = F.cosine_similarity(anchor_emb, candidate_emb,)
                if neg_distance > pos_distance and neg_distance < pos_distance + margin:
                    semi_hard_candidates.append(candidate)
        
        if semi_hard_candidates:
            return random.choice(semi_hard_candidates)
        else:
            return self._select_negative_hard_original(anchor_id, positive_id)
    
    def _select_negative(self, anchor_id, positive_id, margin):
        """Select negative based on mining strategy (optimized)."""
        if self.mining_strategy == 'random':
            return self._select_negative_random(anchor_id, positive_id, margin)
        elif self.mining_strategy == 'hard':
            return self._select_negative_hard_vectorized(anchor_id, positive_id, margin)
        elif self.mining_strategy == 'semi_hard':
            return self._select_negative_semi_hard_vectorized(anchor_id, positive_id, margin)
        else:
            raise ValueError(f"Unknown mining strategy: {self.mining_strategy}")
    
    def __len__(self):
        # Each twin pair generates 2 anchor-positive combinations
        # Each combination generates num_negatives_per_positive triplets
        return len(self.twin_pairs) * 2 * self.num_negatives_per_positive
    
    def __getitem__(self, idx):
        # Decode index to find which twin pair and which direction
        pair_idx = idx // (2 * self.num_negatives_per_positive)
        remainder = idx % (2 * self.num_negatives_per_positive)
        direction = remainder // self.num_negatives_per_positive  # 0 or 1
        negative_idx = remainder % self.num_negatives_per_positive
        
        # Debug: Check if pair_idx is valid
        if pair_idx >= len(self.twin_pairs):
            print(f"Error: pair_idx {pair_idx} >= len(twin_pairs) {len(self.twin_pairs)}")
            print(f"idx: {idx}, pair_idx: {pair_idx}")
            return self.__getitem__(0)  # Fallback to first item
        
        # Get the twin pair
        twin_pair = self.twin_pairs[pair_idx]
        if len(twin_pair) != 2:
            print(f"Error: Expected 2 subjects in twin pair, got {len(twin_pair)}: {twin_pair}")
            return self.__getitem__((idx + 1) % len(self))
            
        subject1_id, subject2_id = twin_pair
        
        # Determine anchor and positive based on direction
        if direction == 0:
            anchor_id, positive_id = subject1_id, subject2_id
        else:
            anchor_id, positive_id = subject2_id, subject1_id
        
        # Select negative
        negative_id = self._select_negative(anchor_id, positive_id, margin=self.margin)
        
        if negative_id is None:
            # Fallback if no valid negative found
            return self.__getitem__((idx + 1) % len(self))
        
        # Load images
        try:
            anchor_img = self._load_image(anchor_id)
            positive_img = self._load_image(positive_id)
            negative_img = self._load_image(negative_id)
        except Exception as e:
            print(f"Error loading images: {e}")
            return self.__getitem__((idx + 1) % len(self))
        
        # Convert to tensors and add channel dimension
        anchor_img = torch.tensor(anchor_img, dtype=torch.float32).unsqueeze(0)
        positive_img = torch.tensor(positive_img, dtype=torch.float32).unsqueeze(0)
        negative_img = torch.tensor(negative_img, dtype=torch.float32).unsqueeze(0)
        
        # Apply transforms
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        
        return {
            'anchor': anchor_img,
            'positive': positive_img, 
            'negative': negative_img,
            'anchor_id': anchor_id,
            'positive_id': positive_id,
            'negative_id': negative_id
        }
    
    def clear_embedding_cache(self):
        """Clear cached embeddings and precomputed matrix."""
        self.cached_embeddings.clear()
        self.embedding_matrix = None
        self.subject_to_idx.clear()
    
    def clear_image_cache(self):
        """Clear cached images to free memory."""
        self.cached_images.clear()
    
    def clear_all_caches(self):
        """Clear both embedding and image caches."""
        self.cached_embeddings.clear()
        self.cached_images.clear()
        self.embedding_matrix = None
        self.subject_to_idx.clear()
    
    def update_embeddings(self):
        """Recompute embeddings after model update."""
        if self.precompute_embeddings and self.model is not None:
            self.clear_embedding_cache()
            self._precompute_all_embeddings()
    
    def get_cache_info(self):
        """Get information about current cache usage."""
        return {
            'cached_images': len(self.cached_images),
            'cached_embeddings': len(self.cached_embeddings),
            'total_subjects': len(self.all_subjects),
            'precomputed_embeddings': self.embedding_matrix is not None,
            'embedding_matrix_shape': self.embedding_matrix.shape if self.embedding_matrix is not None else None
        }
 

class MRI3DAugmentation:
    def __init__(self, p=0.5):
        self.p = p
        
    def random_rotate3d(self, volume):
        if random.random() < self.p:
            # Convert to numpy for rotation
            volume_np = volume.squeeze(0).numpy()  # Remove channel dim for rotation
            axes = random.choice([(0,1), (1,2), (0,2)])
            angle = random.uniform(-20, 20)
            rotated = scipy.ndimage.rotate(volume_np, angle, axes=axes, reshape=False, mode='nearest')
            rotated = torch.from_numpy(rotated).float()
            return rotated.unsqueeze(0)  # Add channel dim back
        return volume
    
    def random_flip(self, volume):
        if random.random() < self.p:
            # Add 1 to axis because channel dimension is at position 0
            axis = random.randint(1, 3)  # Choose from dimensions 1, 2, or 3
            return torch.flip(volume, [axis])
        return volume
    
    def random_intensity_shift(self, volume, max_shift=0.1):
        if random.random() < self.p:
            shift = random.uniform(-max_shift, max_shift)
            return torch.clamp(volume + shift, 0, 1)
        return volume
    
    def random_intensity_scale(self, volume, max_scale=0.1):
        if random.random() < self.p:
            scale = random.uniform(1-max_scale, 1+max_scale)
            return torch.clamp(volume * scale, 0, 1)
        return volume
    
    def random_gaussian_noise(self, volume, max_std=0.02):
        if random.random() < self.p:
            noise = torch.randn_like(volume) * random.uniform(0, max_std)
            return torch.clamp(volume + noise, 0, 1)
        return volume
    
    def __call__(self, volume):
        # Apply augmentations (volume is already normalized in dataset)
        volume = self.random_rotate3d(volume)
        volume = self.random_flip(volume)
        # volume = self.random_intensity_shift(volume)
        volume = self.random_intensity_scale(volume)
        volume = self.random_gaussian_noise(volume)
        
        return volume
