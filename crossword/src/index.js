import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import './styles/main.scss';
import { getCrossword } from './static.js';
import "./styles/patternlock.css"
import Game from './Game';
import * as serviceWorkerRegistration from './serviceWorkerRegistration';



class App extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            crossword: getCrossword()
        }
    }
		
    render() {
        return (
            <div>
                <Game
                    crossword={this.state.crossword}
                    hintLimit={5}
                />
            </div>
        )
    }
}

// ========================================

ReactDOM.render(
    <App />,
    document.getElementById('root')
);

serviceWorkerRegistration.register();
