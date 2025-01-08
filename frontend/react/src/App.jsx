import { Button, ButtonGroup,Input,Stack ,Textarea,Text } from '@chakra-ui/react';
import {sendText} from './services/mainpage.js';
import React from 'react';
import ReactDOM from 'react-dom';
import { useForm } from "react-hook-form"
import {useEffect } from 'react'
import { useState } from 'react'


const App = () => {
  const { register, handleSubmit } = useForm({
    defaultValues: {
      text: ""
    }
  });
  const [val, setVal] = useState(0);

   return (
       <div>
           <h1  >Detect AI Text </h1>
           <form onSubmit={handleSubmit((data) => sendText(data)
               .then(res =>{
                    setVal(res.data.score)
                   }).catch(err => {
                    alert(err)
                    })

               )}>
                <Textarea {...register("text")} />
               <Button type ='submit' colorScheme='blue'> Analize text</Button>
           </form>

           <Text> Change of AI Text : {val *100}   % </Text>
       </div>
       )
}

export default App
