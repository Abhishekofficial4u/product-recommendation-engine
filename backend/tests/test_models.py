import os
import sys
import pytest
import numpy as np

# Add parent directory of api/ to sys.path so 'api' package is discoverable
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import api.model_loader as ml

@pytest.fixture(scope="module", autouse=True)
def setup_models():
    # Load all models once for testing
    ml.load_all_models()

def test_models_loaded():
    assert ml._svd_art is not None, "SVD model did not load"
    assert ml._content_art is not None, "Content TF-IDF model did not load"
    assert "model" in ml._svd_art
    assert "cosine_sim" in ml._content_art

def test_content_similarity_range():
    cosine_sim = ml._content_art["cosine_sim"]
    # Cosine similarity matrix should have values between 0.0 and 1.0 (since TF-IDF vectors are non-negative)
    assert np.all(cosine_sim >= -1e-7)
    assert np.all(cosine_sim <= 1.0 + 1e-7)
    # Diagonal elements should be 1.0
    diag = np.diagonal(cosine_sim)
    assert np.all(np.isclose(diag, 1.0))

def test_content_score_logic():
    # Toy Story = 1, Aladdin and King of Thieves = 422
    # Since they share genres "Animation|Children's|Comedy", similarity should be high (close to 1.0)
    seeds = [1]
    score = ml._content_score(seeds, 422)
    assert score > 0.5
    assert score <= 1.0

    # Non-existent item should return 0.0
    assert ml._content_score(seeds, 999999) == 0.0
    # Empty seeds should return 0.0
    assert ml._content_score([], 422) == 0.0

def test_hybrid_score_logic():
    # SVD prediction is typically on rating scale 1-5
    # Rescaled Content score is also on 1-5
    # Check that hybrid score combines them into 1-5 range
    seeds = [1, 2, 3]
    h_score = ml._hybrid_score(1, 50, seeds)
    assert h_score >= 1.0
    assert h_score <= 5.0

def test_recommendation_filtering():
    # Check that recommendation result does not contain items the user has already rated (seen items)
    rec_result = ml.recommend(user_id=1, top_k=10, model="hybrid")
    items_list = rec_result["items"]
    assert len(items_list) == 10
    
    # Retrieve user 1's seen items in training data
    user_seen = set(ml._train_df[ml._train_df["user_id"] == 1]["item_id"].tolist())
    
    for item in items_list:
        assert item["item_id"] not in user_seen, f"Recommended item {item['item_id']} was already rated by user 1!"
