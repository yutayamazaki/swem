# Contributing Guidelines

## Coding Style

We can check coding style using flake8.

```shell
pip install -r requirements-dev.txt
flake8 swem
mypy swem
```

The above job contains following packages.

- [flake8](http://flake8.pycqa.org)
- [mypy](http://mypy-lang.org/)


## Testing

```shell
# Run tests and measure coverages.
coverage run -m pytest tests

# Output coverage report.
coverage report -m
```
