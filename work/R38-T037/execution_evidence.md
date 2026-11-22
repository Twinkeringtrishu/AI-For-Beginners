# Execution Evidence: R38-T037

**Task**: R38-T037 (7 of 10)  
**Lesson**: `lessons/5-NLP/15-LanguageModeling`  
**Description**: Cosine-similarity lookup — failure and recovery example  

---

## 1. Environment & Tooling Versions

- **Python**: `Python 3.15.2`
- **Git**: `git version 2.55.0`
- **Dependencies**: Python standard library only (no external packages)

---

## 2. Example Run Output

```text
==================================================================
R38-T037: Cosine Similarity Failure & Recovery Pipeline Demo
==================================================================
Step 1 | Status: SUCCESS   | Sim: 1.0   
  Inputs:      vec_a=[1, 0], vec_b=[1, 0]
  Explanation: Calculated valid cosine similarity: 1.0000.

Step 2 | Status: SUCCESS   | Sim: 0.0   
  Inputs:      vec_a=[1, 0], vec_b=[0, 1]
  Explanation: Calculated valid cosine similarity: 0.0000.

Step 3 | Status: UNDEFINED | Sim: None  
  Inputs:      vec_a=[1, 0], vec_b=[0, 0]
  Explanation: Zero vector has norm 0.0; cosine similarity direction is undefined -> returned sentinel (None).

Step 4 | Status: ERROR     | Sim: None  
  Inputs:      vec_a=[1, 0], vec_b=[1, 0, 99]
  Explanation: Invalid/corrupted data: ValueError: Vector dimension mismatch: 2 vs 3

Step 5 | Status: SUCCESS   | Sim: 1.0    [RECOVERED]
  Inputs:      vec_a=[0, 1], vec_b=[0, 1]
  Explanation: Calculated valid cosine similarity: 1.0000.
```

---

## 3. Unit Test Execution Output

```text
test_arbitrary_vector_similarity (test_example.TestCosineSimilarityFailureAndRecovery.test_arbitrary_vector_similarity)
Known 2D vector cosine similarity: [3, 4] vs [4, 3] = 24 / 25 = 0.96. ... ok
test_custom_pipeline_recovery_from_corrupt_type (test_example.TestCosineSimilarityFailureAndRecovery.test_custom_pipeline_recovery_from_corrupt_type)
Pipeline catches TypeError on corrupt input and recovers on the next valid step. ... ok
test_deliberate_wrong_result_fails (test_example.TestCosineSimilarityFailureAndRecovery.test_deliberate_wrong_result_fails)
Verify that a deliberate wrong result makes the test fail. ... ok
test_dimension_mismatch_raises_value_error (test_example.TestCosineSimilarityFailureAndRecovery.test_dimension_mismatch_raises_value_error)
Vectors of differing lengths raise ValueError. ... ok
test_empty_vectors_raise_value_error (test_example.TestCosineSimilarityFailureAndRecovery.test_empty_vectors_raise_value_error)
Empty vectors raise ValueError. ... ok
test_failure_and_recovery_demonstrated (test_example.TestCosineSimilarityFailureAndRecovery.test_failure_and_recovery_demonstrated)
Focus: Exercise a documented invalid/missing-data case, explain the error, ... ok
test_identical_vectors_yield_one (test_example.TestCosineSimilarityFailureAndRecovery.test_identical_vectors_yield_one)
[1,0] versus [1,0] gives 1. ... ok
test_non_finite_values_raise_value_error (test_example.TestCosineSimilarityFailureAndRecovery.test_non_finite_values_raise_value_error)
NaN or infinite values raise ValueError. ... ok
test_non_numeric_elements_raise_type_error (test_example.TestCosineSimilarityFailureAndRecovery.test_non_numeric_elements_raise_type_error)
Booleans, strings, or None values raise TypeError. ... ok
test_opposite_vectors_yield_negative_one (test_example.TestCosineSimilarityFailureAndRecovery.test_opposite_vectors_yield_negative_one)
Opposite direction vectors yield -1.0. ... ok
test_orthogonal_vectors_yield_zero (test_example.TestCosineSimilarityFailureAndRecovery.test_orthogonal_vectors_yield_zero)
[1,0] versus [0,1] gives 0. ... ok
test_zero_vector_returns_undefined_sentinel (test_example.TestCosineSimilarityFailureAndRecovery.test_zero_vector_returns_undefined_sentinel)
A zero vector returns the documented undefined-result sentinel (None). ... ok

----------------------------------------------------------------------
Ran 12 tests in 0.001s

OK
```

---

## 4. Verification Checklist

- [x] `[1,0]` versus `[1,0]` gives `1.0`.
- [x] `[1,0]` versus `[0,1]` gives `0.0`.
- [x] A zero vector returns the documented undefined-result sentinel (`None`).
- [x] Exercised a documented invalid/missing-data case, explained the error, and demonstrated recovery on next valid input.
- [x] Lesson link and example agree on inputs, outputs, and prerequisite tools.
- [x] Contribution focus demonstrated, and deliberate wrong result makes test fail (`AssertionError` verified).
- [x] No unsupported claims about real hardware, cloud service behavior, or trained models.
