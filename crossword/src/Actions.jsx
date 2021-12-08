import React, {useState} from "react";
import { iconList, iconShuffle } from './Svg';

const Actions = ({history, handleShuffle}) => {
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
				    <button onClick={(e) => handleShuffle()} title="Shuffle">
					    { iconShuffle() }
				    </button>
          </div>
			    { renderList && history }
        </>
    );
};
export default Actions;
