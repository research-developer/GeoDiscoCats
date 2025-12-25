# GeoDiscoCats - Geometric Partition Framework Testing

Empirical testing framework for validating category-theoretic properties across different partitions in the Geometric Partition Framework.

## Overview

This project provides a comprehensive test suite for verifying the mathematical foundations of the Geometric Partition Framework, which models software systems as compositions of categorical structures.

## Partitions

The framework tests five core partitions:

1. **Applications Partition** - Traced monoidal category (Kleisli arrows)
2. **Schemas Partition** - Distributive lattice (Heyting elements)
3. **Data Partition** - Statistical manifold with Fisher metric (Data points)
4. **Migrations Partition** - Fundamental groupoid (Patches)
5. **Configurations Partition** - Dependent types (Fiber sections)

## Test Coverage

### Category Theory Laws
- **Kleisli Laws**: Identity and associativity for monadic composition
- **Heyting Algebra**: Lattice operations, distributivity, and intuitionistic implication
- **Groupoid Laws**: Composition, inverses, and path properties
- **Natural Transformations**: Adjunction structure and triangle identities

### Geometric Properties
- **Fisher Metric**: Positive definiteness, symmetry, triangle inequality
- **Embedding Geometry**: Partition clustering and cross-partition distances
- **Information Loss**: Pythagorean comma and translation matrices

## Installation

```bash
# Install base dependencies
pip install -e .

# Install with embedding support
pip install -e ".[embeddings]"

# Install with development tools
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/test_kleisli_laws.py

# Run with coverage
pytest --cov=partition_geometry
```

## Dependencies

- Python >= 3.10
- NumPy >= 1.24.0
- SciPy >= 1.11.0
- pytest >= 7.4.0

## Project Structure

```
GeoDiscoCats/
├── partition_geometry/     # Core module
│   └── __init__.py        # Partition base units documentation
├── tests/                 # Test suite
│   ├── test_kleisli_laws.py
│   ├── test_heyting_laws.py
│   ├── test_fisher_metric_laws.py
│   ├── test_groupoid_laws.py
│   ├── test_natural_transformations.py
│   ├── test_embedding_geometry.py
│   └── conftest.py
└── pyproject.toml        # Project configuration
```

## Mathematical Foundation

The framework validates that software components satisfy rigorous mathematical properties:

- **Functoriality**: Composition preserves structure
- **Adjunctions**: Schema-Data transformations are adjoint functors
- **Metric Properties**: Fisher information geometry on data manifolds
- **Groupoid Structure**: Migration operations form a fundamental groupoid

## Status

✅ All 84 tests passing

## Contributing

This is a research project exploring categorical foundations for software systems. The test suite provides empirical validation of theoretical hypotheses.

## License

This is a research project. License information will be determined based on publication and collaboration requirements.
