"""
Test that the Kleisli arrow structure correctly models Application partition.

Kleisli category laws for monad T:
- Left identity:  return >=> f  =  f
- Right identity: f >=> return  =  f
- Associativity:  (f >=> g) >=> h  =  f >=> (g >=> h)

Additionally test traced monoidal structure for feedback/loops.
"""

import numpy as np
from typing import Callable, TypeVar, Generic, Tuple, List
from dataclasses import dataclass
import pytest

A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')
D = TypeVar('D')


@dataclass
class Result(Generic[A]):
    """Option monad for effectful computation."""
    value: A | None
    success: bool
    trace: List[str]  # Track composition history

    @classmethod
    def pure(cls, a: A) -> 'Result[A]':
        return cls(value=a, success=True, trace=['pure'])

    @classmethod
    def fail(cls, reason: str) -> 'Result[A]':
        return cls(value=None, success=False, trace=[f'fail:{reason}'])

    def bind(self, f: Callable[[A], 'Result[B]']) -> 'Result[B]':
        if not self.success:
            return Result(None, False, self.trace + ['bind_skip'])
        result = f(self.value)
        result.trace = self.trace + ['bind'] + result.trace
        return result


class KleisliArrow(Generic[A, B]):
    """
    A morphism A → Result[B] in the Kleisli category.

    This is the proposed base unit for the Applications partition.
    """

    def __init__(self, f: Callable[[A], Result[B]], name: str = "arrow"):
        self._f = f
        self.name = name

    def __call__(self, a: A) -> Result[B]:
        return self._f(a)

    def compose(self, other: 'KleisliArrow[B, C]') -> 'KleisliArrow[A, C]':
        """Kleisli composition: self >=> other"""
        def composed(a: A) -> Result[C]:
            return self._f(a).bind(other._f)
        return KleisliArrow(composed, f"({self.name} >=> {other.name})")

    @classmethod
    def identity(cls) -> 'KleisliArrow[A, A]':
        """Kleisli identity: return/pure"""
        return cls(lambda a: Result.pure(a), "id")


class TestKleisliLaws:
    """Verify Kleisli category laws hold."""

    @pytest.fixture
    def sample_arrows(self):
        """Create test arrows for numerical computation."""
        f = KleisliArrow(
            lambda x: Result.pure(x * 2) if x < 100 else Result.fail("overflow"),
            "double"
        )
        g = KleisliArrow(
            lambda x: Result.pure(x + 10),
            "add10"
        )
        h = KleisliArrow(
            lambda x: Result.pure(x ** 0.5) if x >= 0 else Result.fail("negative"),
            "sqrt"
        )
        return f, g, h

    def test_left_identity(self, sample_arrows):
        """return >=> f = f"""
        f, _, _ = sample_arrows
        identity = KleisliArrow.identity()

        # identity >=> f
        composed = identity.compose(f)

        test_values = [0, 1, 5, 50, 99, 100, -1]
        for x in test_values:
            direct = f(x)
            via_identity = composed(x)

            assert direct.success == via_identity.success, \
                f"Left identity failed for {x}: success mismatch"
            if direct.success:
                assert direct.value == via_identity.value, \
                    f"Left identity failed for {x}: value mismatch"

    def test_right_identity(self, sample_arrows):
        """f >=> return = f"""
        f, _, _ = sample_arrows
        identity = KleisliArrow.identity()

        # f >=> identity
        composed = f.compose(identity)

        test_values = [0, 1, 5, 50, 99, 100, -1]
        for x in test_values:
            direct = f(x)
            via_identity = composed(x)

            assert direct.success == via_identity.success, \
                f"Right identity failed for {x}"
            if direct.success:
                assert direct.value == via_identity.value, \
                    f"Right identity failed for {x}"

    def test_associativity(self, sample_arrows):
        """(f >=> g) >=> h = f >=> (g >=> h)"""
        f, g, h = sample_arrows

        # (f >=> g) >=> h
        left_assoc = f.compose(g).compose(h)

        # f >=> (g >=> h)
        right_assoc = f.compose(g.compose(h))

        test_values = [0, 1, 4, 9, 16, 25, 50]
        for x in test_values:
            left_result = left_assoc(x)
            right_result = right_assoc(x)

            assert left_result.success == right_result.success, \
                f"Associativity failed for {x}: success mismatch"
            if left_result.success:
                assert np.isclose(left_result.value, right_result.value), \
                    f"Associativity failed for {x}: {left_result.value} != {right_result.value}"

    def test_traced_structure_feedback(self):
        """
        Test traced monoidal structure for feedback loops.

        Trace operation: Tr^U_{A,B}: Hom(A⊗U, B⊗U) → Hom(A, B)
        This models iterative/recursive procedures.
        """
        # Iterative computation: keep applying until fixed point
        def iterate_until_stable(
            f: KleisliArrow[float, float],
            tolerance: float = 1e-6,
            max_iter: int = 100
        ) -> KleisliArrow[float, float]:
            """Trace-like operation: feed output back as input."""
            def traced(x: float) -> Result[float]:
                current = x
                for i in range(max_iter):
                    result = f(current)
                    if not result.success:
                        return result
                    if abs(result.value - current) < tolerance:
                        return Result.pure(result.value)
                    current = result.value
                return Result.fail("no_convergence")
            return KleisliArrow(traced, f"Tr({f.name})")

        # Newton's method for sqrt(2): x -> (x + 2/x) / 2
        newton_step = KleisliArrow(
            lambda x: Result.pure((x + 2/x) / 2) if x > 0 else Result.fail("nonpositive"),
            "newton_sqrt2"
        )

        sqrt2_finder = iterate_until_stable(newton_step)
        result = sqrt2_finder(1.0)

        assert result.success
        assert np.isclose(result.value, np.sqrt(2), atol=1e-6), \
            f"Traced computation should find sqrt(2), got {result.value}"


class TestKleisliSemantics:
    """Test that Kleisli arrows capture procedural/transformational semantics."""

    def test_procedure_encoding(self):
        """
        Procedural discourse should map to Kleisli arrow composition.

        "First double, then add 10, finally take square root"
        → double >=> add10 >=> sqrt
        """
        # Encode procedure steps as arrows
        steps = {
            "double": KleisliArrow(lambda x: Result.pure(x * 2), "double"),
            "add10": KleisliArrow(lambda x: Result.pure(x + 10), "add10"),
            "sqrt": KleisliArrow(
                lambda x: Result.pure(np.sqrt(x)) if x >= 0 else Result.fail("negative"),
                "sqrt"
            )
        }

        # Compose procedure
        procedure = steps["double"].compose(steps["add10"]).compose(steps["sqrt"])

        # Execute
        result = procedure(8)  # 8 → 16 → 26 → √26 ≈ 5.1

        assert result.success
        assert np.isclose(result.value, np.sqrt(26))

        # The composition structure preserves procedural order
        assert "double" in procedure.name
        assert "add10" in procedure.name
        assert "sqrt" in procedure.name

    def test_effect_propagation(self):
        """Effects (failures) should propagate through composition."""
        safe = KleisliArrow(lambda x: Result.pure(x + 1), "safe")
        risky = KleisliArrow(
            lambda x: Result.fail("boom") if x > 5 else Result.pure(x),
            "risky"
        )

        # safe >=> risky >=> safe
        pipeline = safe.compose(risky).compose(safe)

        # Should succeed for small inputs
        assert pipeline(3).success  # 3 → 4 → 4 → 5

        # Should fail and propagate for large inputs
        result = pipeline(5)  # 5 → 6 → FAIL
        assert not result.success
        assert "bind_skip" in result.trace  # Subsequent step was skipped


class TestKleisliExtensions:
    """Test extended Kleisli structure for richer Application semantics."""

    def test_parallel_composition(self):
        """
        Test parallel (tensor) composition: f ⊗ g.

        This models concurrent/parallel execution in procedures.
        """
        f = KleisliArrow(lambda x: Result.pure(x * 2), "double")
        g = KleisliArrow(lambda x: Result.pure(x + 1), "inc")

        # Parallel composition on pairs
        def tensor(
            arr1: KleisliArrow[A, B],
            arr2: KleisliArrow[C, D]
        ) -> KleisliArrow[Tuple[A, C], Tuple[B, D]]:
            def tensored(pair: Tuple[A, C]) -> Result[Tuple[B, D]]:
                a, c = pair
                r1 = arr1(a)
                r2 = arr2(c)
                if r1.success and r2.success:
                    return Result.pure((r1.value, r2.value))
                return Result.fail("parallel_failure")
            return KleisliArrow(tensored, f"({arr1.name} ⊗ {arr2.name})")

        parallel = tensor(f, g)
        result = parallel((5, 10))  # (5, 10) → (10, 11)

        assert result.success
        assert result.value == (10, 11)

    def test_conditional_branching(self):
        """
        Test conditional composition for branching procedures.

        This models if-then-else in procedural discourse.
        """
        branch_positive = KleisliArrow(
            lambda x: Result.pure(x * 2),
            "positive_branch"
        )
        branch_negative = KleisliArrow(
            lambda x: Result.pure(abs(x)),
            "negative_branch"
        )

        def conditional(
            predicate: Callable[[A], bool],
            if_true: KleisliArrow[A, B],
            if_false: KleisliArrow[A, B]
        ) -> KleisliArrow[A, B]:
            def branched(a: A) -> Result[B]:
                if predicate(a):
                    return if_true(a)
                else:
                    return if_false(a)
            return KleisliArrow(branched, f"if-then-else")

        cond = conditional(lambda x: x >= 0, branch_positive, branch_negative)

        assert cond(5).value == 10   # Positive: doubled
        assert cond(-3).value == 3   # Negative: abs


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
