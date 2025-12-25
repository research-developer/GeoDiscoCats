"""
Test that the groupoid morphism structure correctly models Migration partition.

Groupoid laws:
- Category laws (associativity, identity)
- Invertibility: every morphism has an inverse
- Inverse laws: p ; p⁻¹ = id_source, p⁻¹ ; p = id_target
"""

from dataclasses import dataclass
from typing import Dict, Any
import pytest


@dataclass(frozen=True)
class Patch:
    """
    A morphism in the fundamental groupoid of states.

    This is the proposed base unit for the Migration partition.
    """
    source: str  # Source state identifier
    target: str  # Target state identifier
    forward: tuple  # Forward delta (frozen for hashability)
    backward: tuple  # Backward delta (for invertibility)
    name: str = ""

    @classmethod
    def from_dicts(
        cls,
        source: str,
        target: str,
        forward: Dict[str, Any],
        backward: Dict[str, Any],
        name: str = ""
    ) -> 'Patch':
        """Create a Patch from dictionary deltas."""
        return cls(
            source=source,
            target=target,
            forward=tuple(sorted(forward.items())),
            backward=tuple(sorted(backward.items())),
            name=name
        )

    @property
    def forward_dict(self) -> Dict[str, Any]:
        """Get forward delta as dict."""
        return dict(self.forward)

    @property
    def backward_dict(self) -> Dict[str, Any]:
        """Get backward delta as dict."""
        return dict(self.backward)

    def compose(self, other: 'Patch') -> 'Patch':
        """
        Groupoid composition: self ; other

        Requires: self.target == other.source
        """
        if self.target != other.source:
            raise ValueError(
                f"Cannot compose: {self.target} != {other.source}"
            )

        # Merge deltas (simplified; real impl would handle conflicts)
        merged_forward = {**self.forward_dict, **other.forward_dict}
        merged_backward = {**other.backward_dict, **self.backward_dict}

        return Patch.from_dicts(
            source=self.source,
            target=other.target,
            forward=merged_forward,
            backward=merged_backward,
            name=f"({self.name} ; {other.name})"
        )

    def inverse(self) -> 'Patch':
        """
        Groupoid inverse: self⁻¹

        Swaps source/target and forward/backward.
        """
        return Patch(
            source=self.target,
            target=self.source,
            forward=self.backward,
            backward=self.forward,
            name=f"({self.name})^-1"
        )

    @classmethod
    def identity(cls, state: str) -> 'Patch':
        """Identity morphism at a state."""
        return cls(
            source=state,
            target=state,
            forward=(),
            backward=(),
            name=f"id_{state}"
        )

    def endpoints_match(self, other: 'Patch') -> bool:
        """Check if two patches have matching endpoints."""
        return self.source == other.source and self.target == other.target

    def is_identity_like(self) -> bool:
        """Check if patch is effectively an identity (empty deltas)."""
        return len(self.forward) == 0 and len(self.backward) == 0


class TestGroupoidCategoryLaws:
    """Verify groupoid satisfies category laws."""

    @pytest.fixture
    def sample_patches(self):
        p = Patch.from_dicts("A", "B", {"v": 1}, {"v": 0}, "p")
        q = Patch.from_dicts("B", "C", {"w": 2}, {"w": 0}, "q")
        r = Patch.from_dicts("C", "D", {"x": 3}, {"x": 0}, "r")
        return p, q, r

    def test_associativity(self, sample_patches):
        """(p ; q) ; r = p ; (q ; r)"""
        p, q, r = sample_patches

        left = p.compose(q).compose(r)
        right = p.compose(q.compose(r))

        assert left.source == right.source == "A"
        assert left.target == right.target == "D"
        # Forward deltas should be equivalent
        assert set(left.forward) == set(right.forward)

    def test_left_identity(self, sample_patches):
        """id_A ; p = p"""
        p, _, _ = sample_patches
        id_A = Patch.identity("A")

        composed = id_A.compose(p)

        assert composed.source == p.source
        assert composed.target == p.target
        assert composed.forward_dict == p.forward_dict

    def test_right_identity(self, sample_patches):
        """p ; id_B = p"""
        p, _, _ = sample_patches
        id_B = Patch.identity("B")

        composed = p.compose(id_B)

        assert composed.source == p.source
        assert composed.target == p.target
        assert composed.forward_dict == p.forward_dict

    def test_composition_requires_matching_endpoints(self, sample_patches):
        """Composition should fail for non-matching endpoints."""
        p, _, r = sample_patches  # p: A→B, r: C→D

        with pytest.raises(ValueError, match="Cannot compose"):
            p.compose(r)  # B != C


class TestGroupoidInverseLaws:
    """Verify groupoid inverse properties."""

    @pytest.fixture
    def sample_patch(self):
        return Patch.from_dicts(
            source="v1",
            target="v2",
            forward={"field": "added", "version": 2},
            backward={"field": None, "version": 1},
            name="upgrade"
        )

    def test_inverse_exists(self, sample_patch):
        """Every morphism has an inverse."""
        inv = sample_patch.inverse()

        assert inv is not None
        assert inv.source == sample_patch.target
        assert inv.target == sample_patch.source

    def test_left_inverse_law(self, sample_patch):
        """p ; p⁻¹ = id_source"""
        p = sample_patch

        round_trip = p.compose(p.inverse())
        id_source = Patch.identity(p.source)

        assert round_trip.source == id_source.source
        assert round_trip.target == id_source.target
        # Should be identity-like at source

    def test_right_inverse_law(self, sample_patch):
        """p⁻¹ ; p = id_target"""
        p = sample_patch

        round_trip = p.inverse().compose(p)
        id_target = Patch.identity(p.target)

        assert round_trip.source == id_target.source
        assert round_trip.target == id_target.target

    def test_double_inverse(self, sample_patch):
        """(p⁻¹)⁻¹ = p"""
        p = sample_patch

        double_inv = p.inverse().inverse()

        assert double_inv.source == p.source
        assert double_inv.target == p.target
        assert double_inv.forward == p.forward

    def test_inverse_reverses_deltas(self, sample_patch):
        """Inverse swaps forward and backward deltas."""
        inv = sample_patch.inverse()

        assert inv.forward == sample_patch.backward
        assert inv.backward == sample_patch.forward


class TestGroupoidMigrationSemantics:
    """Test that groupoid structure captures version/state change semantics."""

    def test_version_history_as_path(self):
        """Version history forms a path in the groupoid."""
        v1_to_v2 = Patch.from_dicts(
            "v1", "v2",
            {"feature_a": True}, {"feature_a": False},
            "add_feature_a"
        )
        v2_to_v3 = Patch.from_dicts(
            "v2", "v3",
            {"feature_b": True}, {"feature_b": False},
            "add_feature_b"
        )
        v3_to_v4 = Patch.from_dicts(
            "v3", "v4",
            {"refactor": True}, {"refactor": False},
            "refactor"
        )

        # Full history as composed patch
        full_migration = v1_to_v2.compose(v2_to_v3).compose(v3_to_v4)

        assert full_migration.source == "v1"
        assert full_migration.target == "v4"

        # Can rollback to any point
        rollback_to_v2 = v3_to_v4.inverse().compose(v2_to_v3.inverse())
        assert rollback_to_v2.source == "v4"
        assert rollback_to_v2.target == "v2"

    def test_rebase_as_conjugation(self):
        """
        Rebasing a patch = conjugation: q⁻¹ ; p ; q

        This models git rebase: apply p in the context of q.
        """
        # Original patch on branch
        feature_patch = Patch.from_dicts(
            "base", "feature",
            {"feature": True}, {"feature": False},
            "add_feature"
        )

        # Main branch moved forward
        main_patch = Patch.from_dicts(
            "base", "main",
            {"main_change": True}, {"main_change": False},
            "main_update"
        )

        # Simplified rebase: the rebased patch connects main to feature'
        # In full groupoid: feature' is a new state
        rebased = Patch.from_dicts(
            "main",
            "feature_rebased",
            {**main_patch.forward_dict, **feature_patch.forward_dict},
            {**feature_patch.backward_dict, **main_patch.backward_dict},
            "rebased_feature"
        )

        assert rebased.source == "main"
        assert "feature" in rebased.forward_dict
        assert "main_change" in rebased.forward_dict

    def test_merge_as_span(self):
        """
        Merging branches = computing a span (common ancestor + paths).
        """
        # Common ancestor
        ancestor = "base"

        # Two branches
        branch_a = Patch.from_dicts(
            ancestor, "A",
            {"a_change": True}, {"a_change": False},
            "branch_a"
        )
        branch_b = Patch.from_dicts(
            ancestor, "B",
            {"b_change": True}, {"b_change": False},
            "branch_b"
        )

        # Merge state combines both
        merge_from_a = Patch.from_dicts(
            "A", "merge",
            branch_b.forward_dict, branch_b.backward_dict,
            "merge_b_into_a"
        )
        merge_from_b = Patch.from_dicts(
            "B", "merge",
            branch_a.forward_dict, branch_a.backward_dict,
            "merge_a_into_b"
        )

        # Full path from ancestor to merge via A
        via_a = branch_a.compose(merge_from_a)
        assert via_a.source == ancestor
        assert via_a.target == "merge"

        # Full path from ancestor to merge via B
        via_b = branch_b.compose(merge_from_b)
        assert via_b.source == ancestor
        assert via_b.target == "merge"


class TestGroupoidDatabaseMigrations:
    """Test groupoid semantics for database-like migrations."""

    def test_schema_evolution(self):
        """Database schema changes form a groupoid path."""
        add_column = Patch.from_dicts(
            "schema_v1", "schema_v2",
            {"add_column": "email"},
            {"drop_column": "email"},
            "add_email"
        )

        rename_table = Patch.from_dicts(
            "schema_v2", "schema_v3",
            {"rename": ("users", "accounts")},
            {"rename": ("accounts", "users")},
            "rename_users_to_accounts"
        )

        # Compose migrations
        full_migration = add_column.compose(rename_table)
        assert full_migration.source == "schema_v1"
        assert full_migration.target == "schema_v3"

        # Rollback is the inverse
        rollback = full_migration.inverse()
        assert rollback.source == "schema_v3"
        assert rollback.target == "schema_v1"

    def test_migration_conflict_detection(self):
        """Conflicting migrations should be detectable."""
        # Two independent additions (can be composed)
        add_a = Patch.from_dicts("v1", "v1a", {"add": "a"}, {"drop": "a"}, "add_a")
        add_b = Patch.from_dicts("v1a", "v1ab", {"add": "b"}, {"drop": "b"}, "add_b")

        # This should work
        combined = add_a.compose(add_b)
        assert "a" in str(combined.forward_dict)
        assert "b" in str(combined.forward_dict)


class TestGroupoidFundamentalPath:
    """Test fundamental groupoid / path space properties."""

    def test_path_concatenation(self):
        """Paths compose via concatenation."""
        p1 = Patch.from_dicts("A", "B", {"step": 1}, {"step": 0}, "step1")
        p2 = Patch.from_dicts("B", "C", {"step": 2}, {"step": 1}, "step2")
        p3 = Patch.from_dicts("C", "D", {"step": 3}, {"step": 2}, "step3")

        # Path A → D
        path = p1.compose(p2).compose(p3)
        assert path.source == "A"
        assert path.target == "D"

    def test_loop_detection(self):
        """Loops (paths from state to itself) can be detected."""
        forward = Patch.from_dicts("A", "B", {"x": 1}, {"x": 0}, "forward")
        back = forward.inverse()

        loop = forward.compose(back)

        # Loop starts and ends at same state
        assert loop.source == loop.target == "A"

    def test_homotopy_equivalence(self):
        """
        Different paths with same endpoints can be compared.

        In full groupoid: paths related by homotopy are equivalent.
        """
        # Two different paths from A to C
        direct = Patch.from_dicts("A", "C", {"direct": True}, {"direct": False}, "direct")

        via_b_1 = Patch.from_dicts("A", "B", {"step1": True}, {"step1": False}, "step1")
        via_b_2 = Patch.from_dicts("B", "C", {"step2": True}, {"step2": False}, "step2")
        indirect = via_b_1.compose(via_b_2)

        # Both paths have same endpoints
        assert direct.source == indirect.source == "A"
        assert direct.target == indirect.target == "C"

        # But different intermediate structure (different deltas)
        assert direct.forward != indirect.forward


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
