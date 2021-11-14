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

class Guessbox extends React.Component {
    constructor(props) {
        super(props);
        this.handleSubmit = this.handleSubmit.bind(this);
        this.input = React.createRef();
    }

    handleSubmit(event) {
        this.props.onGuess(this.input.current.value);
        this.input.current.value = "";
        event.preventDefault();
    }

    render() {
        return (
            <form onSubmit={this.handleSubmit}>
                <div>
                    {this.props.letters.map((l,i)=>(
                        <span key={i} className="cell available">
                            {l}
                        </span>
                    ))}
                </div>
                <input type="text" ref={this.input} />
                <input type="submit" value="Submit" />
            </form>
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
            for(const [l, pos] of mapWord(hit)) {
                pad[pos] = {
                    l:l,
                    guess:guesses.length
                };
            }
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
            <table className="board">
                <tbody>
                    {this.state.rows.map(y => this.renderRow(y))}
                </tbody>
            </table>
        );
    }

    render() {
        /*
        const history = this.state.history;
        const current = history[history.length - 1];
        const winner = calculateWinner(current.squares);

        const moves = history.map((step, move) => {
            const desc = move ?
                  'Go to move #' + move :
                  'Go to game start';
            return (
                <li>
                    <button onClick={() => this.jumpTo(move)}>{desc}</button>
                </li>
            );
        });

        let status;
        if (winner) {
            status = 'Winner: ' + winner;
        } else {
            status = 'Next player: ' + (this.state.xIsNext ? 'X' : 'O');
        }
        */
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

