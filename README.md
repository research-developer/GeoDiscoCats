# GeoDiscoCats: Geometric Discourse Categories

A test-driven framework for validating the **Geometric Partition Framework** hypothesis that discourse naturally organizes into distinct geometric/categorical partitions with systematic inter-partition translations.

## Overview

This project empirically tests two interconnected hypotheses:

### 1. Base Units Hypothesis
Each discourse partition has a natural "base unit" with clean compositional properties:

| Partition | Base Unit | Geometric Structure |
|-----------|-----------|---------------------|
| **Applications** | Kleisli Arrow | Traced Monoidal Category |
| **Schemas** | Heyting Element | Distributive Lattice |
| **Data** | DataPoint | Statistical Manifold (Fisher Metric) |
| **Migrations** | Patch | Fundamental Groupoid |
| **Configurations** | Fiber Section | Dependent Type |

### 2. Natural Transformation Mediation Hypothesis
Cross-partition translation is mediated by adjoint functors, with the unit (η) and counit (ε) of the adjunction quantifying systematic information loss (analogous to the Pythagorean comma in music theory).

## Installation

```bash
# Clone the repository
git clone https://github.com/research-developer/GeoDiscoCats.git
cd GeoDiscoCats

# Install dependencies
pip install -e .

# Optional: Install embedding support
pip install -e ".[embeddings]"

# Optional: Install development tools
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run tests for specific partition
pytest tests/test_kleisli_laws.py -v
pytest tests/test_heyting_laws.py -v
pytest tests/test_fisher_metric_laws.py -v
pytest tests/test_groupoid_laws.py -v

# Run tests with coverage
pytest tests/ --cov=partition_geometry --cov-report=html
```

## Test Suite Structure

```
tests/
├── test_kleisli_laws.py         # Applications partition (Kleisli arrows)
├── test_heyting_laws.py         # Schemas partition (Heyting algebra)
├── test_fisher_metric_laws.py   # Data partition (Fisher metric)
├── test_groupoid_laws.py        # Migrations partition (Groupoid morphisms)
├── test_natural_transformations.py  # Inter-partition adjunctions
└── test_embedding_geometry.py   # Embedding-based validation
```

### Test Coverage
- **84 tests** validating algebraic laws and compositional properties
- **100% passing** rate
- Coverage of:
  - Category theory laws (identity, associativity, etc.)
  - Lattice laws (meet, join, distributivity)
  - Metric properties (triangle inequality, symmetry)
  - Adjunction structure (unit, counit, triangle identities)
  - Information loss quantification

## Mathematical Foundations

### Category Theory Concepts
- **Kleisli Category**: Models effectful computation (Applications)
- **Traced Monoidal Category**: Models feedback loops in procedures
- **Heyting Algebra**: Models intuitionistic type theory (Schemas)
- **Groupoid**: Models reversible transformations (Migrations)

### Geometric Structures
- **Fisher Information Metric**: Natural Riemannian metric on statistical manifolds
- **Fréchet Mean**: Generalization of arithmetic mean to curved spaces
- **Adjoint Functors**: Formalize the "best approximation" relationship between partitions

### The Pythagorean Comma Analogy
Just as (3/2)^12 ≠ 2^7 in music creates systematic tuning errors, round-trip translations between partitions don't perfectly close. This "comma" structure:
- Is systematic and measurable
- Varies between partition pairs
- Accumulates over multiple translations
- Reflects fundamental information-theoretic constraints

## Examples

### 1. Kleisli Arrows (Applications)
```python
from partition_geometry import KleisliArrow, Result

# Define procedure steps
double = KleisliArrow(
    lambda x: Result.pure(x * 2) if x < 100 else Result.fail("overflow"),
    "double"
)
add_ten = KleisliArrow(lambda x: Result.pure(x + 10), "add_ten")

# Compose procedures
procedure = double.compose(add_ten)

# Execute
result = procedure(5)  # 5 → 10 → 20
assert result.success and result.value == 20
```

### 2. Heyting Elements (Schemas)
```python
from partition_geometry import HeytingElement

# Define types
mammal = HeytingElement(
    required=frozenset(['warm_blooded', 'has_fur']),
    forbidden=frozenset(['cold_blooded'])
)

dog = HeytingElement(
    required=frozenset(['warm_blooded', 'has_fur', 'barks']),
    forbidden=frozenset(['cold_blooded'])
)

# Check subtyping
assert dog <= mammal  # Dog is a subtype of Mammal
```

### 3. Schema-Data Adjunction
```python
from partition_geometry import SchemaDataAdjunction

adjunction = SchemaDataAdjunction()

# Round-trip: Schema → Data → Schema
schema = HeytingElement(required=frozenset(['a', 'b']), forbidden=frozenset(['x']))
recovered, loss = adjunction.unit(schema)

print(f"Information loss: {loss:.2%}")
# Measures what the type system can't express about instances
```

## Project Status

🚧 **In Development** - This is a research prototype

- ✅ Test suite implemented and passing
- ✅ Mathematical foundations validated
- ⚠️ Production code needs to be extracted from tests
- ⚠️ API documentation needed
- ⚠️ Examples and tutorials needed
- ⚠️ Configurations partition not yet implemented

## Contributing

This is a research project. Contributions are welcome, especially:
- Additional partition implementations
- Real-world examples
- Documentation improvements
- Performance optimizations
- Visualization tools

## Research Background

This framework is motivated by observations that different forms of discourse (procedural instructions, type definitions, data instances, version changes, configurations) have distinct mathematical structures that resist unified representation. The goal is to:

1. **Formalize** these distinctions using category theory and differential geometry
2. **Quantify** the information loss in cross-partition translations
3. **Validate** empirically using embeddings and algebraic tests

## Citation

If you use this work in research, please cite:

```bibtex
@software{geodiscocats2025,
  title = {GeoDiscoCats: Geometric Discourse Categories},
  author = {Research Developer},
  year = {2025},
  url = {https://github.com/research-developer/GeoDiscoCats}
}
```

## License

[Add license information]

## References

- **Category Theory**: Mac Lane, "Categories for the Working Mathematician"
- **Heyting Algebra**: van Dalen, "Logic and Structure"
- **Fisher Metric**: Amari & Nagaoka, "Methods of Information Geometry"
- **Adjoint Functors**: Riehl, "Category Theory in Context"

## Contact

For questions or collaboration: [Add contact information]

---

*"Just as music theory needed to accept the Pythagorean comma to make harmony practical, information systems need to accept the translational comma to make cross-domain integration practical."*
