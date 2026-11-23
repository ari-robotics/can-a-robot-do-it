"""ari_sim — turn photos of a workstation into a robot simulation and an engineering diagnostic, on Google DeepMind's robotics plugin."""
import os, sys
# Headless rendering by default on Linux; must be set before mujoco is imported anywhere.
if sys.platform.startswith('linux'): os.environ.setdefault('MUJOCO_GL', 'egl')
__version__ = "0.3.0"
