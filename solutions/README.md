# Worked solutions

One notebook per course notebook, with a worked solution and a discussion for every exercise
(83 in total). The exercise texts are pulled automatically from `../notebooks/build_notebooks.py`, so
questions and solutions always match.

**Try each exercise yourself first.** The learning happens in the attempt; read the solution afterwards
to compare approaches. Most exercises have more than one correct solution.

Many solutions go beyond a single number: they check the answer independently (an exact formula, a
second method, a convergence study) and discuss what the result means for engineering practice - the
habits the course aims to build.

To regenerate and execute all solution notebooks:

```bash
python solutions/build_solutions.py            # all
python solutions/build_solutions.py 05 12      # selected notebooks
```

*For instructors:* if you use the exercises for assessed work, you may prefer to keep the solutions in a
private copy of the repository.
