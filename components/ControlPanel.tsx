import React from 'react';
import { SimulationParams } from '../types.ts';

interface ControlPanelProps {
  params: SimulationParams;
  onChange: (newParams: SimulationParams) => void;
  isHidden: boolean;
  toggleHidden: () => void;
  isPlaying: boolean;
  onTogglePlay: () => void;
  onExport: () => void;
}

export const ControlPanel: React.FC<ControlPanelProps> = ({ 
  params, 
  onChange, 
  isHidden, 
  toggleHidden,
  isPlaying,
  onTogglePlay,
  onExport
}) => {
  const handleChange = (key: keyof SimulationParams, value: number | string) => {
    onChange({ ...params, [key]: value });
  };

  if (isHidden) {
    return (
      <button
        onClick={toggleHidden}
        className="fixed top-4 right-4 z-50 bg-black/60 text-white p-3 rounded-full backdrop-blur-md border border-gray-800 hover:bg-gray-900 transition-all shadow-[0_0_15px_rgba(0,0,0,0.5)]"
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="3"></circle>
          <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
        </svg>
      </button>
    );
  }

  return (
    <div className="fixed top-0 right-0 h-full w-80 bg-black/90 backdrop-blur-md border-l border-gray-900 p-6 overflow-y-auto transition-transform z-40 shadow-2xl">
      <div className="flex justify-between items-center mb-8 pt-2">
        <h2 className="text-lg font-light tracking-widest text-white border-b border-blue-900/50 pb-2 w-full">
          CONTROLLI <span className="text-blue-500 font-bold">CIMATICA</span>
        </h2>
        <button onClick={toggleHidden} className="absolute right-6 top-8 text-gray-500 hover:text-white">
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
        </button>
      </div>

      <div className="space-y-8">
        {/* Time Control Group */}
        <div className="space-y-5">
          <h3 className="text-[10px] uppercase tracking-widest text-green-500 font-bold mb-2">Flusso Temporale</h3>
          
          {/* Play/Pause Button */}
          <button
            onClick={onTogglePlay}
            className={`w-full py-2 px-4 rounded-lg flex items-center justify-center gap-2 transition-all duration-300 border ${
              isPlaying 
                ? 'bg-red-500/10 border-red-500/50 text-red-400 hover:bg-red-500/20' 
                : 'bg-green-500/10 border-green-500/50 text-green-400 hover:bg-green-500/20'
            }`}
          >
            {isPlaying ? (
              <>
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="currentColor" stroke="none"><rect x="6" y="4" width="4" height="16" rx="1" /><rect x="14" y="4" width="4" height="16" rx="1" /></svg>
                <span className="text-xs font-bold tracking-wider">FERMA SIMULAZIONE</span>
              </>
            ) : (
              <>
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="currentColor" stroke="none"><path d="M5 3l14 9-14 9V3z" /></svg>
                <span className="text-xs font-bold tracking-wider">AVVIA SIMULAZIONE</span>
              </>
            )}
          </button>
          
          {/* Export Button (Only valid when paused) */}
           <button
            onClick={onExport}
            disabled={isPlaying}
            className={`w-full py-2 px-4 rounded-lg flex items-center justify-center gap-2 transition-all duration-300 border ${
              isPlaying 
                ? 'bg-gray-800/50 border-gray-800 text-gray-600 cursor-not-allowed' 
                : 'bg-cyan-500/10 border-cyan-500/50 text-cyan-400 hover:bg-cyan-500/20 cursor-pointer'
            }`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="7 10 12 15 17 10"></polyline>
              <line x1="12" y1="15" x2="12" y2="3"></line>
            </svg>
            <span className="text-xs font-bold tracking-wider">
              {isPlaying ? 'PAUSA PER SCARICARE' : 'SCARICA SNAPSHOT'}
            </span>
          </button>


          {/* Simulation Speed Slider */}
          <div className="group mt-4">
            <div className="flex justify-between mb-2">
              <label className="text-xs text-gray-400 group-hover:text-white transition-colors">Velocità (Time Scale)</label>
              <span className="text-xs font-mono text-green-400">{params.simulationSpeed.toFixed(2)}x</span>
            </div>
            <input
              type="range"
              min="0.0"
              max="2.0"
              step="0.05"
              value={params.simulationSpeed}
              onChange={(e) => handleChange('simulationSpeed', parseFloat(e.target.value))}
              className="w-full h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-green-500 hover:accent-green-400"
            />
            <p className="text-[10px] text-gray-500 mt-2 leading-tight">
              Rallenta il tempo per osservare i dettagli delle interferenze in slow motion.
            </p>
          </div>
        </div>

        <div className="h-px bg-gradient-to-r from-transparent via-gray-800 to-transparent" />

        {/* Container Geometry Group */}
        <div className="space-y-5">
          <h3 className="text-[10px] uppercase tracking-widest text-orange-500 font-bold mb-2">Geometria Recipiente</h3>
          
          <div className="group">
            <div className="flex justify-between mb-2">
              <label className="text-xs text-gray-400 group-hover:text-white transition-colors">Diametro (cm)</label>
              <span className="text-xs font-mono text-orange-400">{params.diameter.toFixed(1)} cm</span>
            </div>
            <input
              type="range"
              min="1"
              max="50"
              step="0.5"
              value={params.diameter}
              onChange={(e) => handleChange('diameter', parseFloat(e.target.value))}
              className="w-full h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-orange-500 hover:accent-orange-400"
            />
          </div>

          <div className="group">
            <div className="flex justify-between mb-2">
              <label className="text-xs text-gray-400 group-hover:text-white transition-colors">Profondità Acqua (cm)</label>
              <span className="text-xs font-mono text-orange-400">{params.depth.toFixed(1)} cm</span>
            </div>
            <input
              type="range"
              min="0.5"
              max="30"
              step="0.5"
              value={params.depth}
              onChange={(e) => handleChange('depth', parseFloat(e.target.value))}
              className="w-full h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-orange-500 hover:accent-orange-400"
            />
          </div>
           <div className="flex items-center justify-between">
            <label className="text-xs text-gray-400">Colore Fluido</label>
            <div className="relative overflow-hidden w-8 h-8 rounded-full border border-gray-600">
                <input
                type="color"
                value={params.liquidColor}
                onChange={(e) => handleChange('liquidColor', e.target.value)}
                className="absolute -top-2 -left-2 w-12 h-12 p-0 border-none cursor-pointer"
                />
            </div>
          </div>
        </div>

        <div className="h-px bg-gradient-to-r from-transparent via-gray-800 to-transparent" />

        {/* Wave Physics Group */}
        <div className="space-y-5">
          <h3 className="text-[10px] uppercase tracking-widest text-blue-500 font-bold mb-2">Risonanza Liquida</h3>
          
          <div className="group">
            <div className="flex justify-between mb-2">
              <label className="text-xs text-gray-400 group-hover:text-white transition-colors">Frequenza (Hz)</label>
              <span className="text-xs font-mono text-blue-400">{params.frequency.toFixed(1)}</span>
            </div>
            <input
              type="range"
              min="0"
              max="60"
              step="0.5"
              value={params.frequency}
              onChange={(e) => handleChange('frequency', parseFloat(e.target.value))}
              className="w-full h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-blue-500 hover:accent-blue-400"
            />
          </div>

          <div className="group">
            <div className="flex justify-between mb-2">
              <label className="text-xs text-gray-400 group-hover:text-white transition-colors">Altezza Onda (Ampiezza)</label>
              <span className="text-xs font-mono text-blue-400">{params.amplitude.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.01"
              max="1.0"
              step="0.01"
              value={params.amplitude}
              onChange={(e) => handleChange('amplitude', parseFloat(e.target.value))}
              className="w-full h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-blue-500 hover:accent-blue-400"
            />
          </div>
        </div>

        <div className="h-px bg-gradient-to-r from-transparent via-gray-800 to-transparent" />

        {/* Lighting Group */}
        <div className="space-y-5">
          <h3 className="text-[10px] uppercase tracking-widest text-purple-500 font-bold mb-2">Setup Ottico</h3>

          <div className="group">
            <div className="flex justify-between mb-2">
              <label className="text-xs text-gray-400 group-hover:text-white transition-colors">Altezza Camera (Zoom/Z)</label>
              <span className="text-xs font-mono text-purple-400">{params.cameraHeight.toFixed(1)}</span>
            </div>
            <input
              type="range"
              min="2.0"
              max="30.0"
              step="0.5"
              value={params.cameraHeight}
              onChange={(e) => handleChange('cameraHeight', parseFloat(e.target.value))}
              className="w-full h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-purple-500 hover:accent-purple-400"
            />
          </div>
          
           <div className="group">
            <div className="flex justify-between mb-2">
              <label className="text-xs text-gray-400 group-hover:text-white transition-colors">Densità LED (Totale)</label>
              <span className="text-xs font-mono text-purple-400">{params.ledCount.toFixed(0)}</span>
            </div>
            <input
              type="range"
              min="10"
              max="144"
              step="1"
              value={params.ledCount}
              onChange={(e) => handleChange('ledCount', parseFloat(e.target.value))}
              className="w-full h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-purple-500 hover:accent-purple-400"
            />
          </div>

           <div className="group">
            <div className="flex justify-between mb-2">
              <label className="text-xs text-gray-400 group-hover:text-white transition-colors">Intensità Globale</label>
              <span className="text-xs font-mono text-purple-400">{params.ledBrightness.toFixed(1)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="10.0"
              step="0.1"
              value={params.ledBrightness}
              onChange={(e) => handleChange('ledBrightness', parseFloat(e.target.value))}
              className="w-full h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-purple-500 hover:accent-purple-400"
            />
          </div>

          {/* RING 1 */}
           <div className="group border-l-2 border-purple-500/30 pl-3 mt-4">
            <div className="flex justify-between items-center mb-2">
              <label className="text-[10px] uppercase font-bold text-purple-300">Anello 1 (Principale)</label>
              <div className="relative overflow-hidden w-4 h-4 rounded-full border border-gray-600">
                <input
                type="color"
                value={params.ledColor}
                onChange={(e) => handleChange('ledColor', e.target.value)}
                className="absolute -top-2 -left-2 w-8 h-8 p-0 border-none cursor-pointer"
                />
              </div>
            </div>
            
            <div className="flex justify-between mb-1">
              <label className="text-xs text-gray-500">Altezza</label>
              <span className="text-xs font-mono text-gray-400">{params.ledHeight.toFixed(1)}</span>
            </div>
            <input
              type="range"
              min="0.2"
              max="10.0"
              step="0.1"
              value={params.ledHeight}
              onChange={(e) => handleChange('ledHeight', parseFloat(e.target.value))}
              className="w-full h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-purple-500 hover:accent-purple-400 mb-2"
            />

            <div className="flex justify-between mb-1">
               <label className="text-xs text-gray-500">Raggio</label>
              <span className="text-xs font-mono text-gray-400">{params.ledRadius.toFixed(1)} cm</span>
            </div>
            <input
              type="range"
              min="1.0"
              max="30.0"
              step="0.5"
              value={params.ledRadius}
              onChange={(e) => handleChange('ledRadius', parseFloat(e.target.value))}
              className="w-full h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-purple-500 hover:accent-purple-400 mb-2"
            />
            
             <div className="flex justify-between mb-1">
              <label className="text-xs text-gray-500">Dimensione</label>
              <span className="text-xs font-mono text-gray-400">{params.ledSize.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.05"
              max="2.0"
              step="0.05"
              value={params.ledSize}
              onChange={(e) => handleChange('ledSize', parseFloat(e.target.value))}
              className="w-full h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-purple-500 hover:accent-purple-400"
            />
          </div>

          {/* RING 2 */}
          <div className="group border-l-2 border-pink-500/30 pl-3 mt-4">
            <div className="flex justify-between items-center mb-2">
              <label className="text-[10px] uppercase font-bold text-pink-300">Anello 2 (Secondario)</label>
               <div className="relative overflow-hidden w-4 h-4 rounded-full border border-gray-600">
                <input
                type="color"
                value={params.led2Color}
                onChange={(e) => handleChange('led2Color', e.target.value)}
                className="absolute -top-2 -left-2 w-8 h-8 p-0 border-none cursor-pointer"
                />
              </div>
            </div>
            
            <div className="flex justify-between mb-1">
              <label className="text-xs text-gray-500">Altezza</label>
              <span className="text-xs font-mono text-gray-400">{params.led2Height.toFixed(1)}</span>
            </div>
            <input
              type="range"
              min="0.2"
              max="10.0"
              step="0.1"
              value={params.led2Height}
              onChange={(e) => handleChange('led2Height', parseFloat(e.target.value))}
              className="w-full h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-pink-500 hover:accent-pink-400 mb-2"
            />

            <div className="flex justify-between mb-1">
               <label className="text-xs text-gray-500">Raggio</label>
              <span className="text-xs font-mono text-gray-400">{params.led2Radius.toFixed(1)} cm</span>
            </div>
            <input
              type="range"
              min="1.0"
              max="30.0"
              step="0.5"
              value={params.led2Radius}
              onChange={(e) => handleChange('led2Radius', parseFloat(e.target.value))}
              className="w-full h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-pink-500 hover:accent-pink-400 mb-2"
            />
            
             <div className="flex justify-between mb-1">
              <label className="text-xs text-gray-500">Dimensione</label>
              <span className="text-xs font-mono text-gray-400">{params.led2Size.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.05"
              max="2.0"
              step="0.05"
              value={params.led2Size}
              onChange={(e) => handleChange('led2Size', parseFloat(e.target.value))}
              className="w-full h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-pink-500 hover:accent-pink-400"
            />
          </div>

        </div>
      </div>
      
      <div className="mt-12 text-[10px] text-gray-700 text-center uppercase tracking-widest">
        Physics Engine v2.8
      </div>
    </div>
  );
};