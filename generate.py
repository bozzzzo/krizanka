
import collections
import functools
import itertools
import json
import math
import operator
import os
import platform
import random
import re
import time
import sys
import multiprocessing
import threading


from contextlib import contextmanager


def wizium_root():
    return os.environ.get("WIZIUM_ROOT", "./Wizium")
sys.path.append(wizium_root() + "/Wrappers/Python")

from x import add_to_crossword

# ############################################################################

# Update those paths if needed !
if platform.system()=='Linux':
    PATH = wizium_root() + '/Binaries/Linux/libWizium.so'
elif platform.system()=='Darwin':
    PATH = wizium_root() + '/Binaries/Darwin/liblibWizium.dylib'
else:
    PATH = wizium_root() + '/Binaries/Windows/libWizium_x64.dll'


# ============================================================================
def solve (wiz, max_black=0, heuristic_level=0, seed=0, black_mode='DIAG'):
    """Solve the grid

    wiz             Wizium instance
    max_black       Max number of black cases to add (0 if not allowed)
    heuristic_level Heuristic level (0 if deactivated)
    seed            Random Number Generator seed (0: take at random)
    """
# ============================================================================

    if not seed: seed = random.randint(1, 1000000)

    # Configure the solver
    wiz.solver_start (seed=seed, black_mode=black_mode, max_black=max_black, heuristic_level=heuristic_level)
    tstart = time.time ()

    # Solve with steps of 500ms max, in order to draw the grid content evolution
    while True:
        status = wiz.solver_step (max_time_ms=5000)

        # print (status)

        if status.fillRate == 100:
            ret = True
            break
        if status.fillRate == 0:
            ret = False
            break

    # Ensure to release grid content
    wiz.solver_stop ()

    tend = time.time ()
    #print (status.counter, status.fillRate, "Compute time: {:.01f}s".format (tend-tstart))
    return ret



@contextmanager
def wizium_ctx(*, alphabet):
    # Create a Wizium instance

    from libWizium import Wizium
    dylib = os.path.join (os.getcwd (), PATH)
    assert os.path.exists(dylib)
    print(dylib)
    wiz = Wizium (dylib, alphabet=alphabet)

    yield wiz

    del wiz


def make_crossword(dictionary):

    alphabet = "".join(sorted(functools.reduce(operator.or_, map(set, dictionary)))).upper()

    # print(f"Dictionary with {len(dictionary)} words uses alphabet '{alphabet}'")


    dictionary = sorted(dictionary, key=len)
    dictionary, keyword = dictionary[:-1], dictionary[-1]

    # print(keyword)
    # print(dictionary)

    with wizium_ctx(alphabet=alphabet) as wiz:
        # Load dictionary
        wiz.dic_clear()
        n = wiz.dic_add_entries(dictionary)

        total = sum(map(len, dictionary)) + len(keyword)
        width = min(max(int(math.sqrt(total / 2.5)), len(keyword)), 11)
        height = min(int(total / (width * 2.5)), width)
        size = width * height
        print(width, height)
        wiz.grid_set_size(width, height)

        if width >= len(keyword):
            wiz.grid_write (0,1, keyword, 'H', add_block=True)
        wiz.grid_set_box (0, 0, 'BLACK')
        wiz.grid_set_box (1, 0, 'BLACK')

        for max_black in range(size//5, size//2):
            for _ in range(50):
                if solve(wiz, max_black=max_black, heuristic_level=-1, black_mode='ANY'):
                    break
            else:
                continue
            break
        else:
            print("No solution for ", keyword, [width, height], dictionary)
            return None

        lines = wiz.grid_read()

    print(''.join(lines))

    def transpose(lines):
        return list(f"{l}\n" for l in map(''.join, zip(*map(str.strip, lines))))
    print(''.join(transpose(lines)))

    word_re = re.compile(f"[{alphabet}]{{3,}}")
    def find_words(grid, direction):
        return [direction(match.group(), match.start(), row)
                for row, line in enumerate(grid)
                for match in word_re.finditer(line)
                ]

    def Down(word, y, x):
        return dict(d="Down", w=word, s=[x,y])

    def Across(word, x, y):
        return dict(d="Across", w=word, s=[x,y])

    words = find_words(lines, Across) + find_words(transpose(lines), Down)

    used = collections.Counter(w['w'] for w in words)

    repeats = [(w,c) for w, c in used.items() if c > 1]

    crossword = dict(size=[width, height],
                     words=words,
                     letters=keyword.upper(),
                     unused=[w.upper() for w in dictionary if w not in used])

    if repeats:
        print("Repeated words:", repeats)
    letters = sum(1 for line in lines for c in line if c in alphabet)
    print("solution for ", keyword, [width, height], f"Utilization {letters} / {size} = {letters/size:.0%} repeats:{repeats}")

    return crossword


stopping = False
def stop(pool):
    input()
    global stopping
    stopping = True

def generate():

    with open("dictionaries") as f:
        dicts = tuple(sorted(l.split()) for l in f)

    dicts = random.sample(dicts, len(dicts))

    threading.Thread(target=stop, daemon=True).start()

    with multiprocessing.Pool() as pool:

        for c in pool.imap_unordered(make_crossword, dicts):
            if stopping:
                pool.close()
                return
            if c:
                add_to_crossword(c)



if __name__ == '__main__':
    generate()

