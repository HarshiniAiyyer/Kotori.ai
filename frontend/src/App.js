import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Home from './pages/Home';
import Chat from './pages/Chat';
import Resources from './pages/Resources';
import About from './pages/About';

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Home />} />
        <Route path="chat" element={<Chat />} />
        <Route path="resources" element={<Resources />} />
        <Route path="about" element={<About />} />
      </Route>
    </Routes>
  );
}

export default App;