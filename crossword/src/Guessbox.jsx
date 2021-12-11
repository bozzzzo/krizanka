import React, {useState} from "react";
import PatternSelect from './PatternSelect';
import Actions from "./Actions";

function shuffle(array) {
    array = array.slice();
    for (let i = array.length - 1; i > 0; i--) {
        let j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
}

function lettersToState(letters) {
    return letters.map((x) => ({l: x, used: false}));
}

const Guessbox = ({propsLetters, onGuess, onReload, renderHistory, elapsed}) => {
    const [letters, setLetters] = useState(lettersToState(propsLetters));
    const [value, setValue] = useState("");

    React.useEffect(()=>setLetters(lettersToState(propsLetters)), [propsLetters])

    function handleSubmit(val) {
        if (val.length < 3)
            return;
        onGuess(val);
        clear();
    }

    function handleShuffle() {
        setLetters(shuffle(letters));
    }

    function handleClear() {
        clear();
    }

    function clear() {
        setLetters(letters.map(l => ({...l, used:false})));
        setValue("");
    }

    return (
        <>
            <div>
                <div className="c-word">
                    { value &&
                        <div className="c-word__container">
                            {Array.from(value).map((l,i) => (
                                <span key={i}>{l}</span>
                            ))}
                        </div>

                    }
                </div>
                <React.Fragment key={JSON.stringify(letters)}>
                <PatternSelect
                    letters={letters}
                    onClear={handleClear}
                    setValue={setValue}
                    onSubmit={handleSubmit}
                />
                </React.Fragment>

              <span class="elapsed">{elapsed}</span>
            </div>
          <Actions renderHistory={renderHistory}
                   onShuffle={handleShuffle}
                   onReload={onReload} />
        </>
    );
};

export default Guessbox;
