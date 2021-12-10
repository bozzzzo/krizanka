import React, {useState} from "react";
import { iconList, iconShuffle } from './Svg';

const Actions = ({renderHistory, handleShuffle, handleReload}) => {
    const [renderList, setRenderList] = useState(false);



    function handleRenderList() {
		    setRenderList(!renderList);
	  }


    return(
        <>
          <div style={{width: "100vw"}} className="c-actions">
				    <button onClick={ (e) => handleRenderList() }>
					    { iconList() }
				    </button>
            <button onClick={(e) => handleReload()} title="Reload">
              Reload
            </button>
				    <button onClick={(e) => handleShuffle()} title="Shuffle">
					    { iconShuffle() }
				    </button>
          </div>
			    { renderList && renderHistory() }
        </>
    );
};
export default Actions;
