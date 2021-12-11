import React from "react";
import { getCrossword } from './static.js';
import Game from "./Game"

const directions = {
    Down: start => (i => [start[0], start[1] + i]),
    Across: start => (i => [start[0] + i, start[1]])
};

function mapDirection(word) {
    return directions[word.d](word.s);
}

function mapWord(word) {
    const dir = mapDirection(word);
    const letters = Array.from(word.w);
    const items = letters.map((l,i) => [l, dir(i)]);
    return items;
}

function calculatePad(crossword) {
    let pad = Object();
    for(const word of crossword.words) {
        for(const [l, pos] of mapWord(word)) {
            pad[pos] = {
                l:l,
                guess:null
            };
        }
    }
    return pad;
}

function range(x) {
    let iter = [];
    for(let i = 0; i < x; i++) {
        iter.push(i);
    }
    return iter;
}

function inc(x) {
    return (x||0) + 1;
}

const schema_migrations = {
    // each migration is keyed from older version and must produce either:
    // - the state for the next version, or
    // - null if migration is not possible
    0: (state) => ({...state,
                    schema: 1}),
    1: (state) => ({...state,
                    schema: 2,
                    gameplay: {previous: 0, start: Date.now(), current: 0}}),
};

const state_schema_version = 2;

function validate_schema_migrations() {
    for (const i of range(state_schema_version)) {
        if (i in schema_migrations) {
            continue;
        }
        alert("Missing schema_migrations["+i+"], add it.");
        return false;
    }
    if (Object.keys(schema_migrations).length != state_schema_version) {
        alert("Weird schema_migrations, fix it.");
        return false;
    }
    return true;
}

function stateFromCrossword(crossword) {
    const words = new Set(crossword.words.map((w)=>w.w));
    return {
        // Bump state_schema above when you muck with state.
        // and implement the schema_migration, too
        schema: state_schema_version,
        history: [],
        pad: calculatePad(crossword),
        guesses: [],
        score: { guess: 0, known: 0, miss: 0, repeat:0, words:words.size, unused:crossword.unused.length },
        cols: range(crossword.size[0]),
        rows: range(crossword.size[1]),
        crossword: crossword,
        letters: Array.from(crossword.letters).slice().sort(),
        gameplay: {previous: 0, start: Date.now(), current: 0},
        // see above for state schema changes
    };
}

const STATE_KEY = 'game.state';

function restoreState() {
    const store = window.localStorage;
    if (!store) {
        return undefined;
    }
    const ser = store.getItem(STATE_KEY);
    if (!ser) {
        return undefined;
    }
    if (!validate_schema_migrations()) {
        return undefined;
    }
    let state = JSON.parse(ser);
    while (state && state.schema < state_schema_version) {
        state = schema_migrations[state.schema](state);
    }
    state = {...state,
            gameplay: {
                ...state.gameplay,
                current: 0,
                start: Date.now()}};
    return state;
}

function saveState(state) {
    const store = window.localStorage;
    if (!store) {
        return;
    }
    state = {...state,
             gameplay: {
                 previous: state.gameplay.previous + state.gameplay.current,
                 current: 0
             }};
    const ser = JSON.stringify(state);
    store.setItem(STATE_KEY, ser);
}

class GameLogic extends React.Component {
    constructor(props) {
        super(props);
        this.state = restoreState() || stateFromCrossword(getCrossword());
        setInterval(() => this.handleTick(), 1000);
    }

    handleTick() {
        const state = this.state;
        const elapsed = Math.floor((Date.now() - state.gameplay.start) / 1000);
        if (elapsed != state.gameplay.current) {
            this.setState({
                ...state,
                gameplay: {
                    ...state.gameplay,
                    current: elapsed
                }
            });
        }
    }

    handleReload() {
        const state = stateFromCrossword(getCrossword());
        saveState(state);
        this.setState(state);
    }


    handleGuess(w) {
        const is_w = x => x.w === w;
        const fnot = f => ((...args) => !f(...args));

        let history_entry;
        let history = this.state.history;
        let guesses = this.state.guesses;
        let pad = this.state.pad;
        const score = this.state.score;
        let score_update = {};
        const hits = this.state.crossword.words.filter(is_w);
        if (hits.length > 0) {
            pad = {...this.state.pad};
            guesses = guesses.concat([w]);
            history_entry = {w:w, k:history.length, guess:true};
            score_update = {guess: inc(score.guess)};
            for(const hit of hits) {
                for(const [l, pos] of mapWord(hit)) {
                    pad[pos] = {
                        ...pad[pos],
                        l:l,
                        guess:guesses.length
                    };
                }
            }
        } else {
            const known = this.state.crossword.unused.find(x=>x===w);
            history_entry = {w:w, k:history.length, guess:false, known:!!known};
            if (known) {
                score_update = {known: inc(score.known)};
            } else {
                score_update = {miss: inc(score.miss)};
            }
        }

        const repeat = history.find(is_w);
        if (repeat) {
            history = [
                {
                    ...repeat,
                    repeat: 1 + (repeat.repeat || 0)}
            ].concat(history.filter(fnot(is_w)));
            score_update = {repeat: inc(score.repeat)};
        } else {
            history = [history_entry].concat(history);
        }

        const newState = {
            ...this.state,
            history: history,
            pad: pad,
            guesses: guesses,
            score: {...score, ...score_update}
        };
        saveState(newState);
        this.setState(newState);
    }

    handleHint(x,y) {
        let history = this.state.history;
        let pad = {...this.state.pad};
        let cell = pad[[x,y]];
        const w = ("?" + x + ',' + y);
        history = [{x:x, y:y,
                    k:history.length,
                    hint:cell.l,
                    w: w
                   }].concat(history);
        pad[[x,y]] = {
            ...cell,
            hint: 1 + (cell.hint || 0)
        };
        this.setState({
            ...this.state,
            history: history,
            pad: pad
        });
    }


    render() {

        const state = this.state;
        return (
              <Game
                pad={state.pad}
                cols={state.cols}
                rows={state.rows}
                hintLimit={this.props.hintLimit}
                guesses={state.guesses}
                score={state.score}
                letters={state.letters}
                elapsed={state.gameplay.previous + state.gameplay.current}
                onHint={(x,y) => this.handleHint(x,y)}
                onGuess={(w) => this.handleGuess(w)}
                onReload={() => this.handleReload()}
						    />
        );
    }
}
export default GameLogic;
