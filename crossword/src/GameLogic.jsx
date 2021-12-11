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
    0: (state) => ({...state, schema: 1}), 
};

const state_schema_version = 1;

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
        // see above for state schema changes
    };
}

const STATE_KEY = 'game.state';

function restoreState() {
    const store = window.localStorage;
    if (!store) {
        return;
    }
    const ser = store.getItem(STATE_KEY);
    if (!ser) {
        return;
    }
    if (!validate_schema_migrations()) {
        return;
    }
    let state = JSON.parse(ser);
    while (state && state.schema < state_schema_version) {
        state = schema_migrations[state.schema](state);
    }
    return state;
}

function saveState(state) {
    const store = window.localStorage;
    if (!store) {
        return;
    }
    const ser = JSON.stringify(state);
    store.setItem(STATE_KEY, ser);
}

class GameLogic extends React.Component {
    constructor(props) {
        super(props);
        this.state = restoreState() || stateFromCrossword(getCrossword());
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
        return (
              <Game
                pad={this.state.pad}
                cols={this.state.cols}
                rows={this.state.rows}
                hintLimit={this.props.hintLimit}
                guesses={this.state.guesses}
                score={this.state.score}
                letters={this.state.letters}
                onHint={(x,y) => this.handleHint(x,y)}
                onGuess={(w) => this.handleGuess(w)}
                onReload={() => this.handleReload()}
						    />
        );
    }
}
export default GameLogic;
