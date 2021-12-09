from pybitfield import Bitfield # type: ignore
import collections
import operator
import json
from functools import reduce, lru_cache
import pathlib
import pickle
import logging
import random
import typing as t
import dataclasses
import copy
import abc
import itertools

class BitfieldR(Bitfield):
    def get_bit_positions(self):
        i = self.number_of_element - 1
        b = 2 ** i
        x = self.bitfield
        while b:
            if b&x:
                yield i
            b >>= 1
            i -= 1

    def __repr__(self):
        def sample():
            b = 2 ** (self.number_of_element - 1)
            x = self.bitfield
            while b:
                yield "1" if b&x else "0"
                b >>= 1
        return "".join(sample())

    def __and__(self, other):
        if not isinstance(other, Bitfield):
            return NotImplemented
        return type(self)(min(self.number_of_element, other.number_of_element),
                          bitfield=self.bitfield & other.bitfield)

    def __iand__(self, other):
        if not isinstance(other, Bitfield):
            return NotImplemented
        self.number_of_element = min(self.number_of_element, other.number_of_element)
        self.bitfield &= other.bitfield
        return self

    def __or__(self, other):
        if not isinstance(other, Bitfield):
            return NotImplemented
        return type(self)(max(self.number_of_element, other.number_of_element),
                          bitfield=self.bitfield | other.bitfield)

    def __ior__(self, other):
        if not isinstance(other, Bitfield):
            return NotImplemented
        self.number_of_element = max(self.number_of_element, other.number_of_element)
        self.bitfield |= other.bitfield
        return self

    def __invert__(self):
        return type(self)(self.number_of_element,
                          bitfield = ~self.bitfield)



class Wordlist():
    def __init__(self, wordfreq):
        self.wordfreq = list(wordfreq)
        self.words = {w['w']:w['f'] for w in self.wordfreq}
        self.alphabet = freq = collections.defaultdict(int)
        idx = collections.defaultdict(int)
        bit = 1
        for i, w in enumerate(self.wordfreq):
            for l in w['w']:
                freq[l] += 1
                idx[l] |= bit
            bit += bit
        self.index = {w: ~(BitfieldR(len(self.words), bitfield=k))
                      for w, k in idx.items()}

    def freq(self, word:str) -> int:
        return self.words.get(word, 0)

    def only(self, letters):
        def possible(l, w):
            l = collections.Counter(l)
            l.subtract(w)
            return min(l.values()) >= 0
        wf = self.wordfreq
        bit_positions = reduce(
            operator.__and__,
            (bits
             for letter, bits in self.index.items()
             if letter not in letters)).get_bit_positions()
        return (sorted(letters),
                [w for w,f in sorted(
                    ((w,f) for (w,f) in (
                        (wf[i]['w'],wf[i]['f']) for i in bit_positions)
                     if possible(letters, w)),
                    key=lambda kv: (-kv[1], -len(kv[0]),))])

def words_from_freq(name):
    print("Loading words")
    f = json.load(open(name))
    eng = set(map(str.strip, open("wordlist")))
    slo = set(map(str.strip, open("checked")))
    eng = eng - slo
    w = Wordlist(filter(
        lambda w: (len(w['w'])-1 <= len(set(w['w'])) <= len(w['w'])
                   and len(w['w']) >= 3
                   and w['f'] > 10
                   and w['w'] not in eng),
        f))
    return w

def uptodate(name, cache):
    if not cache.exists():
        return False
    if cache.stat().st_mtime < name.stat().st_mtime:
        return False
    if cache.stat().st_mtime < pathlib.Path(__file__).stat().st_mtime:
        return False
    return True

def load_words(name):
    name = pathlib.Path(name)
    cache = name.with_suffix(".cache")
    if uptodate(name, cache):
        try:
            print("Loading cache")
            with cache.open("rb") as f:
                return pickle.load(f)
        except:
            logging.error("Error loading cache %s", cache, exc_info=True)
    w = words_from_freq(name)
    try:
        with cache.open("wb") as f:
            pickle.dump(w, f)
    except:
        logging.error("Error writing cache %s", cache, exc_info=True)
    return w


Coord = t.Tuple[int, int]
def x(p: Coord) -> int:
    return p[0]

def y(p: Coord) -> int:
    return p[1]

@dataclasses.dataclass(frozen=True)
class Span():
    lo:int
    hi:int

    @classmethod
    def from_(cls, xs: t.Iterable[int]) -> "Span":
        xs = iter(xs)
        first = next(xs)
        return reduce(operator.or_, xs, cls.of(first))

    @classmethod
    def of(cls, x) -> "Span":
        return cls(lo=x,
                   hi=x + 1)

    @classmethod
    def by(cls, x: int, l:int) -> "Span":
        return cls(lo=x,
                   hi=x + l)

    @classmethod
    def zero(cls) -> "Span":
        return cls(lo=0,
                   hi=0)

    def __iter__(self):
        return iter(range(self.lo, self.hi))

    def __or__(self, other: t.Union[int, "Span"]) -> "Span":
        if isinstance(other, int):
            return Span(lo=min(self.lo, other),
                        hi=max(self.hi, other + 1))
        elif isinstance(other, Span):
            return Span(lo=min(self.lo, other.lo),
                        hi=max(self.hi, other.hi))
        else:
            return NotImplemented

    def fits(self, size: int) -> bool:
        return self.hi - self.lo <= size

    def len(self):
        return self.hi - self.lo

@dataclasses.dataclass(frozen=True)
class Rect():
    w: Span
    h: Span

    def __or__(self, other: "Rect") -> "Rect":
        if isinstance(other, Rect):
            return Rect(w=self.w | other.w,
                        h=self.h | other.h)
        else:
            return NotImplemented

    @classmethod
    def zero(cls) -> "Rect":
        return cls(w=Span.zero(), h=Span.zero())

    def fits(self, size: int):
        return self.w.fits(size) and self.h.fits(size)

    def area(self):
        return self.w.len() * self.h.len()


@dataclasses.dataclass(frozen=True)
class Word(abc.ABC):
    word:str
    start:Coord

    @abc.abstractmethod
    def kw(self) -> t.Iterable[t.Tuple[Coord, str]]:
        ...

    @abc.abstractmethod
    def flanks(self) -> t.Iterable[Coord]:
        ...

    @abc.abstractmethod
    def rect(self) -> Rect:
        ...

    @staticmethod
    @abc.abstractmethod
    def offsetby(p: Coord, i: int) -> Coord:
        ...

    def __len__(self):
        return len(self.word)

    @property
    @abc.abstractmethod
    def boundary(self):
        ...

    @property
    @abc.abstractmethod
    def ortho(self):
        ...

    def __hash__(self):
        return hash(self.word)

    def __eq__(self, other):
        if isinstance(other, Word):
            return self.word == other.word
        else:
            return NotImplemented

    def __lt__(self, other: "Word"):
        if isinstance(other, Word):
            return self.word < other.word
        else:
            return NotImplemented

    def json(self, *, shift: Coord = (0, 0)):
        return dict(
            d=self.__class__.__name__,
            w=self.word,
            s=tuple(i - o for i,o in zip(self.start, shift)))

class Down(Word):
    def kw(self):
        yield ((x(self.start), y(self.start) - 1), "/")
        yield from (((x(self.start), _y), l)
                    for _y, l in enumerate(self.word, y(self.start)))
        yield ((x(self.start), y(self.start) + len(self.word)), "/")

    def flanks(self):
        return itertools.product(
            (x(self.start) + xoff for xoff in [-1, 1]),
            (y(self.start) + yoff for yoff in range(len(self.word))))

    def rect(self) -> Rect:
        return Rect(w=Span.of(x(self.start)),
                    h=Span.by(y(self.start), len(self.word)))

    @staticmethod
    def offsetby(p: Coord, i: int) -> Coord:
        return (x(p), y(p) - i)

    @property
    def boundary(self):
        return '|'

    @property
    def ortho(self):
        return '-'

class Across(Word):
    def kw(self):
        yield ((x(self.start) - 1, y(self.start)), "/")
        yield from (((_x, y(self.start)), l)
                    for _x, l in enumerate(self.word, x(self.start)))
        yield ((x(self.start) + len(self.word), y(self.start)), "/")

    def flanks(self):
        return itertools.product(
            (x(self.start) + xoff for xoff in range(len(self.word))),
            (y(self.start) + yoff for yoff in [-1, 1]))

    def rect(self) -> Rect:
        return Rect(w=Span.by(x(self.start), len(self.word)),
                    h=Span.of(y(self.start)))

    @staticmethod
    def offsetby(p: Coord, i: int) -> Coord:
        return (x(p) - i, y(p))

    @property
    def boundary(self):
        return '-'

    @property
    def ortho(self):
        return '|'

DirectionT = t.Type[t.Union[Down, Across]]

@dataclasses.dataclass(frozen=True)
class Pad():
    text: t.Dict[Coord, str] = dataclasses.field(
        default_factory=dict)
    index: t.Dict[str, t.Set[Coord]] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(set))

    def rect(self):
        if self.text:
            return Rect(
                w=Span.from_(x(k) for k,v in self.text.items() if v != '/'),
                h=Span.from_(y(k) for k,v in self.text.items() if v != '/'))
        else:
            return Rect.zero()

    def with_(self, word: Word) -> "Pad":
        text = self.text.copy()
        index = copy.deepcopy(self.index)
        for k, w in word.kw():
            text[k] = w
            index[w].add(k)
        for k in word.flanks():
            if k not in text:
                text[k] = word.boundary
            else:
                if text[k] == word.ortho:
                    text[k] = "/"
        return dataclasses.replace(self,
                                   text=text,
                                   index=index)

    def fits(self, word: Word, size:int) -> bool:
        total = self.rect() | word.rect()
        if not total.fits(size):
            #print(f"cannot fit {word} in {total} = {self.rect()} | {word.rect()}")
            return False
        for k, w in word.kw():
            if self.text.get(k, w) not in (w, word.ortho):
                #print(f"cannot match {word} at {k} '{w}'!='{self.text[k]}'")
                return False
        return True

@dataclasses.dataclass(frozen=True)
class Crossword():
    size: int
    letters: t.List[str]
    unused: t.List[str]
    text: Pad = dataclasses.field(default_factory=Pad)
    words: t.Tuple[Word] = dataclasses.field(default_factory=tuple)

    def json(self, normalize=True):
        r = self.text.rect()
        if normalize:
            shift = (r.w.lo, r.h.lo)
        else:
            shift = (0, 0)
        return dict(
            size=[r.w.len(), r.h.len()],
            letters=self.letters,
            words=[w.json(shift=shift) for w in self.words],
            unused=self.unused[:],
            )

    def place(self, word: str, direction: DirectionT) -> t.Generator["Crossword", None, None]:
        if self.words:
            positions = self._positions(word, direction)
        else:
            positions = self._initial(word, direction)
        unused = self.unused[:]
        unused.remove(word)
        for placed_word in positions:
            yield dataclasses.replace(
                self,
                unused=unused,
                text=self.text.with_(placed_word),
                words=tuple(sorted(self.words + (placed_word,))),
            )

    def _initial(self, word: str, direction: DirectionT) -> t.Generator[Word, None, None]:
        yield direction(word=word,
                        start=(0,0))

    def _positions(self, word: str, direction: DirectionT) -> t.Generator[Word, None, None]:
        for offset, letter in enumerate(word):
            for coord in self.text.index[letter]:
                position = direction(word=word,
                                     start=direction.offsetby(coord, offset))
                if self.text.fits(position, self.size):
                    yield position

    def score(self):
        if self.words:
            return (len(self.words), sum(map(len, self.words)) / self.text.rect().area(),)
        else:
            return (0,)

    def __hash__(self):
        return hash(self.words)

    def __eq__(self, other):
        if isinstance(other, Crossword):
            self.words == other.words
        else:
            return NotImplemented

class Renderer():
    def __init__(self, *, debug=False, raw=True):
        self.letters = {
            "a": "🄰 ",
            "b": "🄱 ",
            "c": "🄲 ",
            "č": "🄲̌ ",
            "d": "🄳 ",
            "e": "🄴 ",
            "f": "🄵 ",
            "g": "🄶 ",
            "h": "🄷 ",
            "i": "🄸 ",
            "j": "🄹 ",
            "k": "🄺 ",
            "l": "🄻 ",
            "m": "🄼 ",
            "n": "🄽 ",
            "o": "🄾 ",
            "p": "🄿 ",
            "r": "🅁 ",
            "s": "🅂 ",
            "š": "🅂̌ ",
            "t": "🅃 ",
            "u": "🅄 ",
            "v": "🅅 ",
            "z": "🅉 ",
            "ž": "🅉̌ ",
            }
        if debug:
            self.letters.update({
            "/": "⧄ ",
            " ": "⧅ ",
            "-": "▷⃞ ",
            "|": "▽⃞ ",
            })
        else:
            blank = "  "
            self.letters.update({
            "/": blank,
            " ": blank,
            "-": blank,
            "|": blank,
        })

        if raw:
            self.letters = {k: f"{k.upper()} "
                            for k in self.letters}
            if not debug:
                self.letters.update((k, "  ") for k in " /-|")





    def render(self, pad: t.Union[Crossword, Pad]) -> str:
        if isinstance(pad, Crossword):
            pad = pad.text
        ret = []
        for y in pad.rect().h:
            row = []
            for x in pad.rect().w:
                row.append(self.letters[pad.text.get((x,y), " ")])
            ret.append("".join(row))
        return "\n".join(ret)


class Solver():

    def solve(self, *, letters: t.List[str], words: t.Dict[str, int], size: int, breadth:int = 50, limit:int = 5) -> t.List[Crossword]:
        candidates = [Crossword(size=size, letters=letters, unused=list(words))]
        done : t.Set[Crossword] = set()
        while len(done) < limit:
            new_candidates : t.List[Crossword] = []
            for candidate in candidates:
                if candidate.unused:
                    if not candidate.words:
                        max_len = max(len(u) for u in candidate.unused)
                        samples = random.choices(
                            [w
                             for w in candidate.unused
                             if len(w) == max_len],
                            k=5)
                        directions = [Across]
                    else:
                        samples = random.choices(
                            candidate.unused,
                            weights=[words[w]
                                     for w in candidate.unused],
                            k=5)
                        directions = [Across, Down]
                    for word in samples:
                        for direction in directions:
                            new_candidates.extend(candidate.place(word, direction))
                else:
                    done.add(candidate)
            if len(new_candidates) > breadth:
                new_candidates = random.sample(new_candidates, breadth)
            if not new_candidates:
                done.update(candidates)
                break
            else:
                candidates = new_candidates
        done = sorted(done, key=Crossword.score)
        return done[-limit:]


words = load_words("freqs.json")

from pprint import pprint
pprint(len(words.words))
keywords = [(word,freq) for word,freq in words.words.items() if 7 <= len(word) <= 9]
print(len(keywords))
#for keyword, freq in random.choices(keywords, k=10):
#    print(words.only(keyword))


def add_to_crossword(c, fn=pathlib.Path("crossword/src/static.js")):
    marker = "/// insert marker ///"
    with fn.open("r") as f:
        static = f.read()
    if marker not in static:
        raise ValueError("No marker in "+fn)
    parts = list(static.partition(marker))
    parts[1:1] = [json.dumps(c, ensure_ascii=False), ",\n"]
    static = "".join(parts)
    tfn = fn.with_suffix(".tmp")
    try:
        with tfn.open("w") as f:
            f.write(static)
        tfn.rename(fn)
    finally:
        if tfn.is_file():
            tfn.unlink()
    print("Added to", fn)


def weight(w):
    return words.freq(w) if len(w) > 3 else 1

def do_some_crosswords():
    w = "banana pogan minomet ata repa avto omara klobasa slalom pingvin gonoreja".split()
    k, _ = random.choice(keywords)
    _, w = words.only(k)
    print (_, w)
    c= Solver().solve(letters=k,
                      words={i:weight(i) for i in w},
                      size=10,
                      breadth=100,
                      limit=1)

    for last in c:
        print(Renderer(debug=False).render(last))
        add_to_crossword(last.json())


def fish_for_wordlists():
    with open("dictionaries", "w") as f:
        for k, _ in keywords:
            _, w = words.only(k)
            f.write(" ".join(w))
            f.write("\n")
            print(k)

if __name__ == '__main__':
    fish_for_wordlists()

#  with open("check.csv", "w") as fd:
#      import csv
#      writer = csv.writer(fd)
#      writer.writerows(x[:1] for x in sorted(((w,f) for w,f in words.words.items()),
#                              key=lambda x:x[::-1]))
