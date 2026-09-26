# Contributing

Corrections, new exercises, new engineering examples and new methods are welcome.

```bash
git clone https://github.com/ktwyw/engmath && cd engmath
pip install -e ".[dev]"
pytest                                   # unit tests + doctests
python docs/validate.py                  # accuracy checks
python notebooks/build_notebooks.py      # rebuild and execute the notebooks
```

Guidelines

- Library code should stay **readable first**: plain NumPy, short functions, a docstring stating the
  method, its order of accuracy and its limitations.
- Every new method needs a validation check in `docs/validate.py` against SciPy, an exact solution or
  its theoretical convergence order.
- Notebooks are edited in `notebooks/build_notebooks.py`, never directly in the `.ipynb` files. Check
  that every statement in the text matches the executed output.
- Run `ruff check .` before opening a pull request.
