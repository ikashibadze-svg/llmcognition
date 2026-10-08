from llmcognition.evaluation import auc, paired_test


def test_auc_ties_and_extremes():
    assert auc([(0.8, 1), (0.4, 0)]) == 1.0
    assert auc([(0.4, 1), (0.8, 0)]) == 0.0
    assert auc([(0.5, 1), (0.5, 0)]) == 0.5
    assert auc([(0.7, 1), (0.8, 1)]) is None


def test_paired_ci_contains_large_positive_effect():
    a = {str(i): 1.0 for i in range(30)}
    b = {str(i): 0.0 for i in range(30)}
    result = paired_test(a, b, iterations=1000)
    assert result["n_paired"] == 30
    assert result["difference_mean"] == 1.0
    assert result["bootstrap_95_ci"] == [1.0, 1.0]
    assert result["paired_permutation_p_two_sided"] < .01
