import React from "react";
import { version } from '../package.json';
import Guessbox from './Guessbox';
import Pad from "./Pad";

class Game extends React.Component {
    renderHistory() {
        function renderLi(w, i, a) {
            return (
                <span key={w.k}
                      className={((w.guess) ? "guess" :
                                  ((w.known) ? "known" : "weird"))}>
                  {w.w} {w.repeat &&
                         <span>
                           {"➰".repeat(w.repeat)}
                         </span>}
                  {(i < a.length - 1) && ", "}
                </span>);
        }
        return (
            <div className="c-history">
              <p className="known">
                {this.props.history.filter((w)=>w.known && !w.guess).map(renderLi)}
              </p>
              <hr/>
              <p className="unknown">
                {this.props.history.filter((w)=>!w.known && !w.guess).map(renderLi)}
              </p>
              <span className="app-version">version: { version }</span>
            </div>
        );}


    render() {
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
                renderHistory={() => this.renderHistory()}
                elapsed={this.props.elapsed} />
            </div>
        );
    }
}
export default Game;
