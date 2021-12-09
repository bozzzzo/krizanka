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

const Guessbox = ({propsLetters, history, onGuess, handleReload}) => {
    const [letters,setLetters] = useState(propsLetters.map((x) => ({l: x, used: false})));
    const [value, setValue] = useState("");

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
                    handleClear={handleClear}
                    setValue={setValue}
                    handleSubmit={handleSubmit}
                />
                </React.Fragment>


            </div>
          <Actions history={history}
                   handleShuffle={handleShuffle}
                   handleReload={handleReload} />
        </>
    );
};

export default Guessbox;
