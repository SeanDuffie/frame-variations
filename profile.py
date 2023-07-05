import cProfile

from generate_variants import ImgMod

pr = cProfile.Profile()
pr.run('ImgMod()')
