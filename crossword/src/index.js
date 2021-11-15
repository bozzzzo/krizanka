import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';

const directions = {
    Down: start => (i => [start[0], start[1] + i]),
    Across: start => (i => [start[0] + i, start[1]])
};

function mapDirection(word) {
    return directions[word.d](word.s)
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

function Square(props) {
    return (
        <button className="square" onClick={props.onClick}>
            {props.value}
        </button>
    );
}

function shuffle(array) {
    array = array.slice();
    for (let i = array.length - 1; i > 0; i--) {
        let j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
}

class Guessbox extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            letters: props.letters.map((x) => ({l: x, used: false})),
            value: ""
        }
    }

    handleSubmit() {
        if (this.state.value.length < 3)
            return;
        this.props.onGuess(this.state.value);
        this.clear();
    }

    handleShuffle() {
        this.setState({
            ...this.state,
            letters: shuffle(this.state.letters)
        })
    }

    handleClear() {
        this.clear();
    }

    handleDelete() {
        const last = this.state.value.slice(-1);
        const letter = this.state.letters.find(l => (l.used && l.l == last))
        this.setState({
            ...this.state,
            letters: this.state.letters.map(l => (l === letter) ? {...l, used:false} : l),
            value: this.state.value.slice(0, -1)
        })
    }

    clear() {
        this.setState({
            ...this.state,
            letters: this.state.letters.map(l => ({...l, used:false})),
            value: ""
        })
    }

    handleClick(i) {
        if (this.state.letters[i].used)
            return;
        const letters = this.state.letters.slice();
        letters[i].used = true;
        const value = this.state.value + letters[i].l
        this.setState({
            ...this.state,
            letters: letters,
            value: value
        })
    }

    renderLetter(l, i) {
        if (l.used) {
            return (
                <span key={i}
                      className="cell used">
                    {l.l}
                </span>
            );
        } else {
            return (
                <span key={i}
                      className="cell available"
                      onClick={(e) => this.handleClick(i)}>
                    {l.l}
                </span>
            );
        }
    }

    render() {
        return (
            <div className="guessbox">
                <div className="word">
                    <span key="x"
                          className="letter">
                        :
                    </span>
                    {Array.from(this.state.value).map((l,i) => (
                        <span key={i}
                              className="letter">
                            {l}
                        </span>
                    ))}
                    <button onClick={(e)=>this.handleDelete()}>⏪</button>
                </div>
                <div className="chooser">
                    {this.state.letters.map((l,i) => this.renderLetter(l, i))}
                </div>
                <div className="actions">
                    <button onClick={(e) => this.handleShuffle()}>
                        Shuffle
                    </button>
                    <button onClick={(e) => this.handleClear()}>
                        Clear
                    </button>
                    <button onClick={(e) => this.handleSubmit()}
                            className="go">
                        Go
                    </button>
                </div>
            </div>
        );
    }
}



function range(x) {
    let iter = [];
    for(let i = 0; i < x; i++) {
        iter.push(i);
    }
    return iter;
}

class Game extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            history: [],
            pad: calculatePad(props.crossword),
            guesses: [],
            cols: range(props.crossword.size[0]),
            rows: range(props.crossword.size[1]),
            crossword: props.crossword
        };
    }

    /*
    handleClick(i) {
        const history = this.state.history;
        const current = history[history.length - 1];
        const squares = current.squares.slice();
        if (calculateWinner(squares) || squares[i]) {
            return;
        }
        squares[i] = this.state.xIsNext ? 'X' : 'O';
        this.setState({
            history: history.concat([{
                squares: squares
            }]),
            xIsNext: !this.state.xIsNext,
        });
    }
    */

    handleGuess(w) {
        let history = this.state.history;
        let guesses = this.state.guesses;
        let pad = this.state.pad;
        const hit = this.props.crossword.words.find(x=>x.w===w);
        if (hit) {
            pad = {...this.state.pad}
            guesses = guesses.concat([w])
            history = history.concat([w])
            for(const [l, pos] of mapWord(hit)) {
                pad[pos] = {
                    l:l,
                    guess:guesses.length
                };
            }
        } else {
            history = history.concat([w])
        }
        const newState = {
            ...this.state,
            history: history,
            pad: pad,
            guesses: guesses
        };
        console.log(w, hit, newState);
        this.setState(newState);
    }

    renderCell(x,y) {
        let p = this.state.pad[[x,y]];
        let k = x;
        let c;
        let l;
        if (p === undefined) {
            c="cell void";
            l="";
        } else if (p.guess === null) {
            c="cell empty";
            l=" ";
        } else if (p.guess < this.state.guesses.length) {
            c="cell solved";
            l=p.l;
        } else {
            c="cell solved guessed";
            l=p.l;
        }
        return (
            <td key={k}>
                <div className={c}>
                    <span>{l}</span>
                </div>
            </td>
        );
    }

    renderRow(y) {
        return (
            <tr className="board-row" key={y}>
                {this.state.cols.map(x => this.renderCell(x,y))}
            </tr>
        );
    }

    renderPad() {
        return (
            <div className="board">
                <table className="board">
                    <tbody>
                        {this.state.rows.map(y => this.renderRow(y))}
                    </tbody>
                </table>
            </div>
        );
    }

    render() {
        const letters = Array.from(this.props.crossword.letters).slice().sort();
        return (
            <div className="game">
                {this.renderPad()}
                <br/>
                <div className="guessbox">
                    <Guessbox
                        letters={letters}
                        onGuess={(w) => this.handleGuess(w)}
                    />
                </div>
                <div className="history">
                    <ul>
                        {this.state.history.map((w,i)=>(
                            <li key={i}>{w}</li>))}
                    </ul>
                </div>
            </div>
        );
    }
}

class App extends React.Component {
    constructor(props) {
        super(props);
        let crossword = {
            "size": [10, 11],
            "letters": "nemudoma",
            "words": [
                {"d": "Across", "w": "amd", "s": [4, 8]},
                {"d": "Across", "w": "amen", "s": [6, 9]},
                {"d": "Down", "w": "ane", "s": [2, 1]},
                {"d": "Down", "w": "dam", "s": [4, 7]},
                {"d": "Across", "w": "dan", "s": [1, 5]},
                {"d": "Across", "w": "dano", "s": [1, 1]},
                {"d": "Across", "w": "dno", "s": [6, 1]},
                {"d": "Across", "w": "dom", "s": [7, 5]},
                {"d": "Down", "w": "doma", "s": [8, 0]},
                {"d": "Down", "w": "domu", "s": [4, 0]},
                {"d": "Down", "w": "don", "s": [5, 3]},
                {"d": "Down", "w": "duo", "s": [6, 1]},
                {"d": "Down", "w": "med", "s": [7, 3]},
                {"d": "Down", "w": "menda", "s": [3, 3]},
                {"d": "Across", "w": "nad", "s": [2, 7]},
                {"d": "Down", "w": "ned", "s": [1, 3]},
                {"d": "Down", "w": "nem", "s": [2, 7]},
                {"d": "Across", "w": "nemudoma", "s": [1, 3]},
                {"d": "Down", "w": "oda", "s": [6, 7]},
                {"d": "Across", "w": "oman", "s": [6, 7]},
                {"d": "Down", "w": "ona", "s": [8, 5]},
                {"d": "Across", "w": "one", "s": [0, 8]}],
            "unused": ["modem", "domen", "moda", "ano", "emu",
                       "domena", "duma", "mona", "dona", "eon",
                       "mena", "onda", "demon", "amon", "nomad",
                       "meno", "neo", "dao", "omen", "neum", "mone",
                       "medo", "oma", "moden", "memo", "meda",
                       "umen", "nom", "moa", "mamon", "omamen",
                       "noma", "mond", "mun", "neuma"]};
        this.state = {
            crossword: crossword
        }
    }

    render() {
        return (
            <Game
                crossword={this.state.crossword}
            />
        )
    }
}

// ========================================

ReactDOM.render(
    <App />,
    document.getElementById('root')
);
