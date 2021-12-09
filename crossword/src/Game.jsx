import React from "react";
import { version } from '../package.json';
import Guessbox from './Guessbox';
import Pad from "./Pad";

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
    return (x||0) + 1
}

class Game extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            history: [],
            pad: calculatePad(props.crossword),
            guesses: [],
            score: { guess: 0, known: 0, missed: 0, repeat:0 },
            cols: range(props.crossword.size[0]),
            rows: range(props.crossword.size[1]),
            crossword: props.crossword
        };
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
        const hits = this.props.crossword.words.filter(is_w);
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
            const known = this.props.crossword.unused.find(x=>x===w);
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
            score: {...score, ...score_upadte}
        };
        console.log(w, hits, newState);
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

    renderHistory() {
    	  return (
            <div className="c-history">
				      <ul>
					      {this.state.history.map((w,i)=>(
						        <li key={w.k}
								        className={((w.guess) ? "guess" :
									                  ((w.known) ? "known" : "weird"))}>
							        {w.w} {w.repeat &&
						                 <span>
                               {"➰".repeat(w.repeat)}
                             </span>}
						        </li>))}
						    <li className="app-version">version: { version }</li>
				      </ul>
			      </div>
        );
		}

    render() {
        console.log(this.props.crossword);
        const letters = Array.from(this.props.crossword.letters).slice().sort();
        return (
            <div style={{display: "flex", flexDirection:"column", justifyContent:"space-between", height: "100vh"}} className="game">
              <Pad
                pad={this.state.pad}
                cols={this.state.cols}
                rows={this.state.rows}
                hintLimit={this.props.hintLimit}
                handleHint={(x,y) => this.handleHint(x,y)}
                guesses={this.state.guesses}
                score={this.state.score} />
              <Guessbox
                propsLetters={letters}
                onGuess={(w) => this.handleGuess(w)}
						    history={this.renderHistory()} />
            </div>
        );
    }
}
export default Game;
