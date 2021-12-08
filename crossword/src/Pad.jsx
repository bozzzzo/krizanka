import React from "react";

const Pad = ({pad, rows, cols, hintLimit, handleHint, guesses}) => {
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
            opt.onClick = e => handleHint(x,y);
            l=" ";
        } else if (p.guess < guesses.length) {
            opt.className="cell solved";
            l=p.l;
        } else {
            opt.className="cell solved guessed";
            l=p.l;
        }

        if (p && p.hint && p.hint >= hintLimit) {
            opt.className += "cell hint";
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
            <div className="grid">
				{rows.map(y => renderRow(y))}
            </div>
        );
    }

    return (
        <>
            {renderPad()}
        </>
    );
};

export default Pad;
