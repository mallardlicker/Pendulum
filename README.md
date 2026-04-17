--------------------------------------------------------------------------------------

Physics Simulations

--------------------------------------------------------------------------------------

Quick coding of derived ODEs: basically building on functionality along the way, so not the most perfect implementations of rendered shapes etc., but enough to be generally clear in the main file.

Uses the [Pyglet](https://github.com/pyglet/pyglet) library ([installation instructions](https://pyglet.readthedocs.io/en/latest/programming_guide/installation.html)).

Ensure [Pyglet](https://github.com/pyglet/pyglet) is installed, as well as python3, and run the main file to view the simulations (you may also need common libraries such as [Scipy](https://github.com/scipy/scipy)).
```bash
python3 InteractPhysicsSim.py
```
Once open, the numbers keys select which simulation is running, clicking and dragging forces the red mass, the `[` and `]` keys hide/unhide the physical system and tracers respectively, and `Esc` closes the window. Make sure you do press the `[` and `]` keys--they change how you'll view the output of the physical system (and are quite beautiful).