import React from "react";

const Pad = ({pad, rows, cols, hintLimit, onHint, guesses, score}) => {
    function renderCell(x,y) {
        let p = pad[[x,y]];
        let k = x;
        let l;
        let opt = {};
        if (p === undefined) {
            opt.className="cell void";
            l="";
        } else if (p.guess === null && (p.hint || 0) < hintLimit) {
            opt.className="cell empty";
            opt.onClick = e => onHint(x,y);
            l=" ";
        } else if (p.guess === null && (p.hint || 0) >= hintLimit) {
            opt.className="cell";
            l=p.l;
        } else if (p.guess < guesses.length) {
            opt.className="cell solved";
            l=p.l;
        } else {
            opt.className="cell solved guessed";
            l=p.l;
        }

        if (p && p.hint && p.hint >= hintLimit) {
            opt.className += " hint";
        }

        return (
					<div {...opt} key={k}>
							<span>{l}</span>
					</div>
        );
    }

    function renderRow(y) {
        return (
            <div className="row" key={y}>
                {cols.map(x => renderCell(x,y))}
            </div>
        );
    }

    function renderPad() {
    	// todo converted from table
        return (
            <>
              <div className="grid">
				        {rows.map(y => renderRow(y))}
              </div>
              <div className="score">
                <span className="guess"> g {score.guess}/{score.words} </span>
                <span className="known"> k {score.known}/{score.unused} </span>
                <span className="miss"> m {score.miss} </span>
                <span className="repeat"> r {score.repeat} </span>
              </div>
            </>
        );
    }

    return (
        <>
            {renderPad()}
        </>
    );
};

export default Pad;
