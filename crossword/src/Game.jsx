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

    isRepeatGuess(w) {
        let history = this.state.history;
        const isw = x => x.w === w;
        const fnot = f => ((...args) => !f(...args));
        const repeat = history.find(isw);
        let score = this.state.score;
        if (repeat) {
            history = [
                {
                    ...repeat,
                    repeat: 1 + (repeat.repeat || 0)}
            ].concat(history.filter(fnot(isw)));
            score = {...score, repeat: inc(score.repeat)};
            this.setState({
                ...this.state,
                history: history,
                score: score
            });
            return true;
        }
        return false;
    }

    handleGuess(w) {
        if (this.isRepeatGuess(w)) {
            return;
        }
        let history = this.state.history;
        let guesses = this.state.guesses;
        let pad = this.state.pad;
        let score = this.state.score;
        const hit = this.props.crossword.words.find(x=>x.w===w);
        if (hit) {
            pad = {...this.state.pad};
            guesses = guesses.concat([w]);
            history = [{w:w, k:history.length, guess:true}].concat(history);
            score = {...score, guess: inc(score.guess)};
            for(const [l, pos] of mapWord(hit)) {
                pad[pos] = {
                    ...pad[pos],
                    l:l,
                    guess:guesses.length
                };
            }
        } else {
            const known = this.props.crossword.unused.find(x=>x===w);
            history = [{
                w:w,
                k:history.length,
                guess:false,
                known:!!known}
                      ].concat(history);
            if (known) {
                score = {...score, known: inc(score.known)};
            } else {
                score = {...score, miss: inc(score.miss)};
            }
        }
        const newState = {
            ...this.state,
            history: history,
            pad: pad,
            guesses: guesses,
            score: score
        };
        console.log(w, hit, newState);
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
