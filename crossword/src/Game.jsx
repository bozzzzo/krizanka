import React from "react";
import { version } from '../package.json';
import Guessbox from './Guessbox';
import Pad from "./Pad";

class Game extends React.Component {
    renderHistory() {
    	  return (
            <div className="c-history">
				      <ul>
					      {this.props.history.map((w,i)=>(
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
        return (
            <div style={{display: "flex", flexDirection:"column", justifyContent:"space-between", height: "100vh"}} className="game">
              <Pad
                pad={this.props.pad}
                cols={this.props.cols}
                rows={this.props.rows}
                hintLimit={this.props.hintLimit}
                onHint={this.props.onHint}
                guesses={this.props.guesses}
                score={this.props.score} />
              <Guessbox
                propsLetters={this.props.letters}
                onGuess={this.props.onGuess}
                onReload={this.props.onReload}
						    renderHistory={() => this.renderHistory()} />
            </div>
        );
    }
}
export default Game;
