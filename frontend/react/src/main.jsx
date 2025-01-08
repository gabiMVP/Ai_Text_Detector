import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import { ChakraProvider } from '@chakra-ui/react'
import { createStandaloneToast } from '@chakra-ui/react'

const { ToastContainer } = createStandaloneToast()

createRoot(document.getElementById('root')).render(
    <ChakraProvider>
      <StrictMode>
        <App />
      </StrictMode>,
    <ToastContainer />
     </ ChakraProvider>
)
