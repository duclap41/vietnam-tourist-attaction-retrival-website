# reranking_feedback.py
import torch
import os
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class FeedbackManager:
    def __init__(self):
        # First, initialize basic attributes
        self.weights_path = 'weights/'
        self.features_path = './features/'
        os.makedirs(self.weights_path, exist_ok=True)

        # Initialize tracking lists and dictionaries
        self.deleted_images = []
        self.reranked_positions = {}

        # Initialize hyperparameters
        self.negative_lr = 0.1
        self.positive_lr = 0.05
        self.weight_decay = 0.001

        # Then load features and weights
        try:
            # Load features
            self.features_dict = self._load_features()
            self._print_features_info()  # Debug info

            # Initialize weights
            self.weights = {
                'vit': self._init_weights('vit', 768),
                'resnet': self._init_weights('resnet', 2048)
            }

            logger.info("FeedbackManager initialized successfully")

        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            # Ensure basic structures exist even if loading fails
            self.features_dict = {'vit': {}, 'resnet': {}}
            self.weights = {
                'vit': torch.ones(768),
                'resnet': torch.ones(2048)
            }

    def _init_weights(self, model_type, dim):
        """Initialize or load existing weights"""
        try:
            weight_file = os.path.join(self.weights_path, f'{model_type}_weights.pt')
            if os.path.exists(weight_file):
                weights = torch.load(weight_file)
                logger.info(f"Loaded existing weights for {model_type}")
                return weights
            logger.info(f"Initializing new weights for {model_type}")
            return torch.ones(dim)
        except Exception as e:
            logger.error(f"Error initializing weights for {model_type}: {e}")
            return torch.ones(dim)

    def _load_features(self):
        """Load and validate feature dictionaries"""
        features = {}
        try:
            for model in ['vit', 'resnet']:
                file_path = os.path.join(self.features_path, f'{model}_dataset_features.pt')
                if os.path.exists(file_path):
                    loaded_features = torch.load(file_path)

                    # Ensure features are in dictionary format
                    if isinstance(loaded_features, torch.Tensor):
                        features[model] = {str(i): feat for i, feat in enumerate(loaded_features)}
                    elif isinstance(loaded_features, dict):
                        features[model] = loaded_features
                    else:
                        logger.warning(f"Unexpected feature type for {model}: {type(loaded_features)}")
                        features[model] = {}

                    logger.info(f"Loaded {model} features: {len(features[model])} images")
                else:
                    logger.warning(f"Feature file not found: {file_path}")
                    features[model] = {}
        except Exception as e:
            logger.error(f"Error loading features: {e}")
            features = {'vit': {}, 'resnet': {}}
        return features

    def _print_features_info(self):
        """Print debug information about loaded features"""
        for model, features in self.features_dict.items():
            logger.info(f"\nModel: {model}")
            logger.info(f"Features type: {type(features)}")

            # Check if features is dictionary
            if isinstance(features, dict):
                logger.info(f"Number of images: {len(features)}")
                if len(features) > 0:
                    sample_key = next(iter(features))
                    logger.info(f"Sample key type: {type(sample_key)}")
                    logger.info(f"Sample key: {sample_key}")
                    logger.info(f"Sample value type: {type(features[sample_key])}")
            # Check if features is tensor
            elif isinstance(features, torch.Tensor):
                logger.info(f"Tensor shape: {features.shape}")
                logger.info(f"Tensor device: {features.device}")
                logger.info(f"Tensor dtype: {features.dtype}")
            else:
                logger.warning(f"Unknown features type: {type(features)}")

    def save_weights(self, model_type):
        """Save weights to file"""
        try:
            weight_file = os.path.join(self.weights_path, f'{model_type}_weights.pt')
            torch.save(self.weights[model_type], weight_file)
            logger.info(f"Saved weights for {model_type}")
        except Exception as e:
            logger.error(f"Error saving weights: {e}")

    def update_weights_from_deletion(self, img_path, model_type):
        """Update weights based on deleted image"""
        try:
            img_name = img_path.split('/')[-1]
            logger.debug(f"Processing deletion for image: {img_name}")

            # Ensure we're working with a dictionary
            features = self.features_dict[model_type]
            if not isinstance(features, dict):
                logger.error(f"Features for {model_type} is not a dictionary")
                return False

            if img_name in features:
                img_features = features[img_name]
                if not isinstance(img_features, torch.Tensor):
                    img_features = torch.tensor(img_features)

                logger.debug(f"Weights before update - Mean: {self.weights[model_type].mean():.4f}")

                feature_importance = torch.abs(img_features)
                self.weights[model_type] -= self.negative_lr * feature_importance
                self.deleted_images.append(img_name)

                self.normalize_weights(model_type)
                self.save_weights(model_type)

                logger.debug(f"Weights after update - Mean: {self.weights[model_type].mean():.4f}")
                return True
            else:
                logger.warning(f"Image features not found for: {img_name}")
                return False

        except Exception as e:
            logger.error(f"Error in update_weights_from_deletion: {e}")
            return False

    def update_weights_from_reranking(self, reranked_images, model_type):
        """Update weights based on reranking"""
        try:
            updates_made = False
            for img_path, new_rank in reranked_images.items():
                img_name = img_path.split('/')[-1]
                if img_name in features:
                    img_features = features[img_name]
                    if not isinstance(img_features, torch.Tensor):
                        img_features = torch.tensor(img_features)

                    rank_weight = 1.0 / (new_rank + 1)
                    feature_importance = torch.abs(img_features)
                    self.weights[model_type] += self.positive_lr * rank_weight * feature_importance
                    updates_made = True

            if updates_made:
                self.normalize_weights(model_type)
                self.save_weights(model_type)
                logger.info("Successfully updated weights from reranking")
            return updates_made

        except Exception as e:
            logger.error(f"Error in update_weights_from_reranking: {e}")
            return False

    def normalize_weights(self, model_type):
        """Normalize weights"""
        try:
            self.weights[model_type] *= (1 - self.weight_decay)
            self.weights[model_type] = torch.nn.functional.normalize(self.weights[model_type], p=2, dim=0)
            self.weights[model_type] = torch.abs(self.weights[model_type])
        except Exception as e:
            logger.error(f"Error in normalize_weights: {e}")

    def apply_weights_to_query(self, query_features, model_type):
        """Apply weights to query vector"""
        try:
            if not isinstance(query_features, torch.Tensor):
                query_features = torch.tensor(query_features)
            weighted_query = query_features * self.weights[model_type]
            return torch.nn.functional.normalize(weighted_query, p=2, dim=0)
        except Exception as e:
            logger.error(f"Error in apply_weights_to_query: {e}")
            return query_features

    def reset_state(self):
        """Reset the feedback manager state"""
        self.deleted_images = []
        self.reranked_positions = {}
        logger.info("Reset FeedbackManager state")

    def get_state(self):
        """Get current state for debugging"""
        return {
            'deleted_images_count': len(self.deleted_images),
            'reranked_positions_count': len(self.reranked_positions),
            'vit_weights_stats': {
                'mean': float(self.weights['vit'].mean()),
                'std': float(self.weights['vit'].std())
            },
            'resnet_weights_stats': {
                'mean': float(self.weights['resnet'].mean()),
                'std': float(self.weights['resnet'].std())
            }
        }


# Initialize global instance
feedback_manager = FeedbackManager()

# Print initial state
logger.info("Initial FeedbackManager state:")
logger.info(feedback_manager.get_state())