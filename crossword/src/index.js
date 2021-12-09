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
            crossword: {"size": [10, 10], "words": [{"d": "Across", "w": "EKI", "s": [0, 0]}, {"d": "Across", "w": "IŠČE", "s": [6, 0]}, {"d": "Across", "w": "KOPALIŠČE", "s": [0, 1]}, {"d": "Across", "w": "ILA", "s": [0, 2]}, {"d": "Across", "w": "ČEKA", "s": [6, 2]}, {"d": "Across", "w": "PIKA", "s": [0, 3]}, {"d": "Across", "w": "LEPAK", "s": [5, 3]}, {"d": "Across", "w": "KLI", "s": [3, 4]}, {"d": "Across", "w": "ČAO", "s": [1, 5]}, {"d": "Across", "w": "KOPAL", "s": [5, 5]}, {"d": "Across", "w": "ŠOK", "s": [0, 6]}, {"d": "Across", "w": "POČEK", "s": [4, 6]}, {"d": "Across", "w": "EKIPA", "s": [0, 7]}, {"d": "Across", "w": "AKEL", "s": [6, 7]}, {"d": "Across", "w": "OKA", "s": [3, 8]}, {"d": "Across", "w": "ILO", "s": [7, 8]}, {"d": "Across", "w": "APEL", "s": [0, 9]}, {"d": "Down", "w": "EKIPA", "s": [0, 0]}, {"d": "Down", "w": "ŠEPA", "s": [0, 6]}, {"d": "Down", "w": "KOLI", "s": [1, 0]}, {"d": "Down", "w": "ČOK", "s": [1, 5]}, {"d": "Down", "w": "IPAK", "s": [2, 0]}, {"d": "Down", "w": "AKI", "s": [2, 5]}, {"d": "Down", "w": "AKO", "s": [3, 3]}, {"d": "Down", "w": "POL", "s": [3, 7]}, {"d": "Down", "w": "ŠLK", "s": [4, 0]}, {"d": "Down", "w": "PAK", "s": [4, 6]}, {"d": "Down", "w": "LIKO", "s": [5, 3]}, {"d": "Down", "w": "IŠČE", "s": [6, 0]}, {"d": "Down", "w": "OČA", "s": [6, 5]}, {"d": "Down", "w": "ŠČEP", "s": [7, 0]}, {"d": "Down", "w": "PEKI", "s": [7, 5]}, {"d": "Down", "w": "ČEKA", "s": [8, 0]}, {"d": "Down", "w": "AKEL", "s": [8, 5]}, {"d": "Down", "w": "AKEL", "s": [9, 2]}, {"d": "Down", "w": "LOČ", "s": [9, 7]}], "letters": "KOPALIŠČE", "unused": ["AKEL", "AKI", "AKO", "ALI", "ALK", "ALO", "ALPE", "APEL", "API", "EIA", "EKI", "EKIPA", "EKO", "EPIK", "EPIKA", "ILA", "ILKA", "ILO", "IPAK", "IŠČE", "KAL", "KALI", "KALIČ", "KALIŠČE", "KALO", "KAP", "KAPE", "KAPIČ", "KAPO", "KAČ", "KEP", "KEPA", "KEČ", "KEČAP", "KEŠ", "KILA", "KIP", "KIČ", "KIŠA", "KLAP", "KLEP", "KLEPAČ", "KLEČ", "KLEČA", "KLEŠČ", "KLI", "KLIP", "KLIŠE", "KLO", "KLOP", "KLOPA", "KLOPE", "KLOŠČ", "KOL", "KOLA", "KOLAČ", "KOLI", "KOLIČ", "KOLIŠČE", "KOP", "KOPA", "KOPAL", "KOPALIŠČE", "KOPAČ", "KOPEL", "KOPIČ", "KOPIŠČE", "KOČ", "KOČA", "KOŠ", "KOŠIČ", "LAIK", "LAK", "LAKI", "LAKIČ", "LAŠKI", "LAŠČ", "LEK", "LEP", "LEPA", "LEPAK", "LEPŠA", "LEČA", "LEČI", "LEČO", "LEŠ", "LEŠKI", "LIK", "LIKO", "LIPA", "LIPE", "LIČKA", "LIŠP", "LIŠPA", "LOK", "LOKA", "LOP", "LOPA", "LOČ", "LOČEK", "LOŠKI", "LOŠČ", "OKA", "OKEL", "OKLEP", "OLI", "OLIKA", "OPA", "OPAL", "OPEKA", "OPEKAČ", "OPEL", "OPEČI", "OPLA", "OČA", "OČAK", "OČALI", "OČE", "OČEK", "OČI", "OČKA", "PAK", "PAKI", "PAKO", "PALČEK", "PAČ", "PAČI", "PAŠ", "PAŠKI", "PEK", "PEKA", "PEKAČ", "PEKI", "PEKLA", "PEKLO", "PEL", "PELO", "PEČ", "PEČA", "PEČAK", "PEČAL", "PEČI", "PEČKA", "PEŠ", "PEŠKA", "PEŠKI", "PIK", "PIKA", "PIKE", "PIKL", "PIL", "PILA", "PILKA", "PILO", "PIČEL", "PIČLO", "PIŠ", "PIŠEK", "PIŠKA", "PIŠKE", "PIŠČAL", "PIŠČE", "PLAČ", "PLAŠČ", "PLAŠČEK", "PLEČ", "PLEČA", "PLEŠA", "PLEŠKA", "PLOŠČA", "PLOŠČEK", "POK", "POKA", "POKAL", "POL", "POLA", "POLEČI", "POLI", "POLIČ", "POLIČEK", "POLK", "POLKA", "POLKE", "POČ", "POČEK", "POŠLE", "ČAK", "ČAO", "ČAP", "ČEK", "ČEKA", "ČELA", "ČELIKA", "ČELO", "ČEP", "ČEPA", "ČEŠ", "ČEŠKI", "ČIK", "ČIL", "ČILE", "ČIP", "ČIPKA", "ČOK", "ČOKA", "ČOP", "ČOPA", "ČOPKA", "ŠAL", "ŠALE", "ŠAP", "ŠEPA", "ŠIK", "ŠILO", "ŠIPA", "ŠIPEK", "ŠIPKA", "ŠLEK", "ŠLK", "ŠOK", "ŠOLA", "ŠOP", "ŠOPA", "ŠOPEK", "ŠPELA", "ŠPIK", "ŠPIKA", "ŠPIL", "ŠPILA", "ŠČAP", "ŠČEK", "ŠČEP", "ŠČIP", "ŠČIPA"]}
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
