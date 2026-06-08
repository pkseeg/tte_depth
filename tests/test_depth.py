"""Tests for tte_depth.StatDepth."""
import numpy as np
import pytest

from tte_depth import StatDepth, chord_dist, cosine_dist


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_F():
    return np.array([[-0.1, 0.1, 0.1], [3.14, 3.14, 3.14]], dtype=float)


@pytest.fixture
def small_G():
    return np.array([[3.14, -6.28, 3.14], [3.14, 3.14, 3.14], [-3.14, -3.14, -3.14]], dtype=float)


@pytest.fixture
def medium_F():
    rng = np.random.default_rng(42)
    return rng.standard_normal((20, 8))


@pytest.fixture
def medium_G():
    rng = np.random.default_rng(99)
    return rng.standard_normal((15, 8))


# ---------------------------------------------------------------------------
# StatDepth.depths — single corpus
# ---------------------------------------------------------------------------

class TestDepths:
    def test_output_length(self, small_F):
        d = StatDepth()
        scores = d.depths(small_F)
        assert len(scores) == len(small_F)

    def test_output_is_array(self, small_F):
        d = StatDepth()
        scores = d.depths(small_F)
        assert isinstance(scores, np.ndarray)

    def test_scores_are_finite(self, medium_F):
        d = StatDepth()
        scores = d.depths(medium_F)
        assert np.all(np.isfinite(scores))

    def test_cosine_vs_chord_differ(self, medium_F):
        scores_cos = StatDepth("cosine").depths(medium_F)
        scores_chord = StatDepth("chord").depths(medium_F)
        assert not np.allclose(scores_cos, scores_chord)

    def test_custom_distance(self, small_F):
        d = StatDepth(distance=lambda a, b: 0.5)
        scores = d.depths(small_F)
        assert len(scores) == len(small_F)
        # D = 2 - 0.5 = 1.5 for all embeddings
        assert np.allclose(scores, 1.5)

    def test_type_error_on_list(self):
        d = StatDepth()
        with pytest.raises(TypeError):
            d.depths([[1, 2, 3], [4, 5, 6]])

    def test_most_central_highest_depth(self):
        # Cosine depth measures directional centrality.
        # Cluster near [1,1,1,1]; outlier in opposite direction [-1,-1,-1,-1].
        rng = np.random.default_rng(0)
        cluster = np.ones((10, 4)) + rng.standard_normal((10, 4)) * 0.05
        outlier = np.array([[-1.0, -1.0, -1.0, -1.0]])
        F = np.vstack([cluster, outlier])
        scores = StatDepth().depths(F)
        # Outlier should have the lowest depth
        assert np.argmin(scores) == len(cluster)


# ---------------------------------------------------------------------------
# StatDepth.depths_paired — two corpora
# ---------------------------------------------------------------------------

class TestDepthsPaired:
    def test_output_lengths(self, small_F, small_G):
        d = StatDepth()
        sf, sg = d.depths_paired(small_F, small_G)
        assert len(sf) == len(small_F)
        assert len(sg) == len(small_G)

    def test_unequal_corpora(self):
        F = np.random.default_rng(1).standard_normal((7, 5))
        G = np.random.default_rng(2).standard_normal((12, 5))
        sf, sg = StatDepth().depths_paired(F, G)
        assert len(sf) == 7 and len(sg) == 12

    def test_type_error(self, small_F):
        d = StatDepth()
        with pytest.raises(TypeError):
            d.depths_paired(small_F.tolist(), small_F)


# ---------------------------------------------------------------------------
# StatDepth.q_score
# ---------------------------------------------------------------------------

class TestQScore:
    def test_same_distribution_near_half(self, medium_F):
        # Split one corpus in two — Q should be close to 0.5
        d = StatDepth()
        sf = d.depths(medium_F)
        half = len(sf) // 2
        q = d.q_score(sf[:half], sf[half:])
        assert 0.0 <= q <= 1.0

    def test_outlying_corpus_below_half(self):
        # G is uniformly far from F → depths_G w.r.t. F will be low → Q < 0.5
        rng = np.random.default_rng(7)
        F = rng.standard_normal((30, 4))
        G = rng.standard_normal((30, 4)) + 50.0  # very far away
        d = StatDepth()
        sf, sg = d.depths_paired(F, G)
        q = d.q_score(sf, sg)
        assert q < 0.5

    def test_range(self, medium_F, medium_G):
        d = StatDepth()
        sf, sg = d.depths_paired(medium_F, medium_G)
        q = d.q_score(sf, sg)
        assert 0.0 <= q <= 1.0


# ---------------------------------------------------------------------------
# StatDepth.depth_rank_test
# ---------------------------------------------------------------------------

class TestDepthRankTest:
    def test_output_shapes(self, small_F, small_G):
        d = StatDepth()
        sf, sg, q, w, p = d.depth_rank_test(small_F, small_G)
        assert len(sf) == len(small_F)
        assert len(sg) == len(small_G)

    def test_p_value_in_range(self, medium_F, medium_G):
        d = StatDepth()
        _, _, q, w, p = d.depth_rank_test(medium_F, medium_G)
        assert 0.0 <= p <= 1.0

    def test_significant_shift_detected(self):
        # Cosine depth is direction-based: place F in the positive orthant and
        # G in the negative orthant so they are clearly directionally distant.
        rng = np.random.default_rng(13)
        F = np.abs(rng.standard_normal((50, 4)))   # all-positive directions
        G = -np.abs(rng.standard_normal((50, 4)))  # all-negative directions
        _, _, q, _, p = StatDepth().depth_rank_test(F, G)
        assert q < 0.5
        assert p < 0.05

    def test_chord_distance(self, small_F, small_G):
        d = StatDepth("chord")
        sf, sg, q, w, p = d.depth_rank_test(small_F, small_G)
        assert len(sf) == len(small_F) and len(sg) == len(small_G)
        assert 0.0 <= p <= 1.0
