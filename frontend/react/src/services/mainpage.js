import axios from 'axios';


export const sendText = async(text) =>{
    try{
        return await axios.post(
        `${import.meta.env.VITE_API_BASE_URL}/detect_ai_text`,
        text
        )
    }catch(e){
    console.log(e.response)
       throw e
    }
}
//
//export const saveCustomer = async(customer) =>{
//    try{
//        return await axios.post(
//        `${import.meta.env.VITE_API_BASE_URL}/api/v1/customers`,
//        customer
//
//        )
//    }catch(e){
//       throw e
//    }
//}
