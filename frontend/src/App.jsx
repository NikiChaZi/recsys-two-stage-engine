import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Film, Heart, Sliders, Zap, Clock, Database, 
  RefreshCw, Dice5, AlertCircle, X, Orbit,
  Activity, BarChart3, Terminal
} from 'lucide-react';
import Galaxy3D from './components/Galaxy3D';

const API_BASE = 'http://localhost:8000';

// Разворачиваем артикли: "Godfather: Part II, The" -> "The Godfather: Part II"
function normalizeTitle(rawTitle) {
  if (!rawTitle) return '';
  let title = rawTitle.replace(/\s*\(\d{4}\)$/, '').trim();
  const match = title.match(/^(.*),\s*(The|A|An)$/i);
  if (match) {
    title = `${match[2]} ${match[1]}`;
  }
  // Точное название оригинального фильма 1971 года (без Тома и Джерри)
  if (title.toLowerCase().includes('willy wonka')) {
    return 'Willy Wonka & the Chocolate Factory';
  }
  return title.replace(/[:]/g, '').trim();
}

// 2-уровневый загрузчик постеров: OMDb (IMDb) -> Wikipedia
const fetchPosterUrl = async (movie) => {
  const cleanTitle = normalizeTitle(movie.title);

  // ЖЕЛЕЗНЫЙ ФИКС ДЛЯ ВОНКИ 1971 ГОДА (Джин Уайлдер):
  if (cleanTitle.toLowerCase().includes('wonka')) {
    const geneWilderPoster = "https://upload.wikimedia.org/wikipedia/en/thumb/7/7f/WillyWonkaMoviePoster.jpg/500px-WillyWonkaMoviePoster.jpg";
    localStorage.setItem(`poster_${movie.movie_id}`, geneWilderPoster);
    return geneWilderPoster;
  }

  const cacheKey = `poster_${movie.movie_id}`;
  const cached = localStorage.getItem(cacheKey);
  if (cached && cached !== 'null') return cached;

  // 1. Попытка через OMDb (официальный IMDb)
  try {
    const omdbUrl = `https://www.omdbapi.com/?apikey=f7c004c&t=${encodeURIComponent(cleanTitle)}&y=${movie.year || ''}`;
    const res = await axios.get(omdbUrl, { timeout: 3500 });
    if (res.data?.Poster && res.data.Poster !== 'N/A' && !res.data.Title.toLowerCase().includes('tom and jerry')) {
      localStorage.setItem(cacheKey, res.data.Poster);
      return res.data.Poster;
    }
  } catch (e) {}

  // 2. Попытка без указания года
  try {
    const omdbUrl2 = `https://www.omdbapi.com/?apikey=f7c004c&t=${encodeURIComponent(cleanTitle)}`;
    const res2 = await axios.get(omdbUrl2, { timeout: 3500 });
    if (res2.data?.Poster && res2.data.Poster !== 'N/A' && !res2.data.Title.toLowerCase().includes('tom and jerry')) {
      localStorage.setItem(cacheKey, res2.data.Poster);
      return res2.data.Poster;
    }
  } catch (e) {}

  // 3. Запасной вариант: Wikipedia
  try {
    const wikiUrl = `https://en.wikipedia.org/w/api.php?action=query&generator=search&gsrsearch=${encodeURIComponent(cleanTitle + ' film')}&gsrlimit=1&prop=pageimages&pithumbsize=600&format=json&origin=*`;
    const res = await axios.get(wikiUrl, { timeout: 3500 });
    const pages = res.data?.query?.pages;
    if (pages) {
      const firstPage = Object.values(pages)[0];
      const imgUrl = firstPage?.thumbnail?.source;
      if (imgUrl && !imgUrl.toLowerCase().includes('tom_and_jerry')) {
        localStorage.setItem(cacheKey, imgUrl);
        return imgUrl;
      }
    }
  } catch (e) {}

  return null;
};

function MoviePoster({ movie }) {
  const [posterUrl, setPosterUrl] = useState(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let isMounted = true;
    fetchPosterUrl(movie).then(url => {
      if (isMounted && url) setPosterUrl(url);
    });
    return () => { isMounted = false; };
  }, [movie.movie_id]);

  return (
    <div className="absolute inset-0 z-0 overflow-hidden bg-zinc-950">
      {posterUrl ? (
        <img
          src={posterUrl}
          alt={normalizeTitle(movie.title)}
          onLoad={() => setLoaded(true)}
          className={`w-full h-full object-cover object-center transition-all duration-700 ${
            loaded ? 'opacity-65 group-hover:opacity-85 group-hover:scale-105' : 'opacity-0'
          }`}
        />
      ) : (
        <div className="w-full h-full bg-gradient-to-t from-black via-zinc-900 to-zinc-950 opacity-80" />
      )}
      <div className="absolute inset-0 bg-gradient-to-t from-[#09090c] via-[#09090c]/60 to-transparent" />
    </div>
  );
}

export default function App() {
  const [userId, setUserId] = useState(42);
  const [diversity, setDiversity] = useState(0.4);
  const [exploration, setExploration] = useState(0.1);
  const [genreFilter, setGenreFilter] = useState('');
  
  const [viewMode, setViewMode] = useState('feed');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [likedMovies, setLikedMovies] = useState(new Set());
  
  const [selectedMovie, setSelectedMovie] = useState(null);
  const [logs, setLogs] = useState([]);

  const addLog = (msg) => {
    const time = new Date().toLocaleTimeString();
    setLogs(prev => [`[${time}] ${msg}`, ...prev.slice(0, 9)]);
  };

  const fetchRecommendations = async () => {
    setLoading(true);
    setError(null);
    try {
      const params = {
        k: 10,
        diversity: diversity,
        exploration_rate: exploration,
      };
      if (genreFilter) params.genre = genreFilter;

      const res = await axios.get(`${API_BASE}/recommend/${userId}`, { params });
      setData(res.data);
      addLog(`RECSYS: user #${userId} → Qdrant (${res.data.stage1_retrieval_ms}ms) + CatBoost (${res.data.stage2_ranking_ms}ms) = ${res.data.total_latency_ms}ms`);
    } catch (err) {
      setError(err.response?.data?.detail || 'Не удалось связаться с сервером');
      addLog(`ERR: Ошибка связи с API`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRecommendations();
    addLog(`INIT: Платформа готова. Субъект: User #${userId}`);
  }, [userId, genreFilter]);

  const handleLike = async (e, movie) => {
    e.stopPropagation();
    try {
      setLikedMovies(prev => new Set(prev).add(movie.movie_id));
      const cleanName = normalizeTitle(movie.title);
      addLog(`EVENT_STREAM: Отправка клика "${cleanName}" в Redis Stream...`);
      
      await axios.post(`${API_BASE}/feedback`, {
        user_id: Number(userId),
        movie_id: Number(movie.movie_id),
        rating: 5.0,
      });

      addLog(`REDIS: Сессионный вектор User #${userId} адаптирован`);
      fetchRecommendations();
    } catch (err) {
      addLog(`ERR: Не удалось записать событие в Redis`);
    }
  };

  return (
    <div className="min-h-screen bg-[#08080a] text-zinc-100 flex flex-col font-sans selection:bg-amber-500/30 selection:text-amber-200">
      
      {/* Шапка */}
      <header className="border-b border-zinc-800/80 bg-[#0d0d11]/80 backdrop-blur-xl sticky top-0 z-40 px-6 py-3.5">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-amber-500 to-amber-700 flex items-center justify-center shadow-lg shadow-amber-500/20 border border-amber-400/40">
              <Film className="w-5 h-5 text-black stroke-[2.5]" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-black tracking-widest text-zinc-100 uppercase font-mono">
                  PulseRec <span className="text-amber-500">//</span> Engine
                </h1>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-zinc-800 text-zinc-400 border border-zinc-700/60">
                  v2.5M
                </span>
              </div>
              <p className="text-[11px] text-zinc-400 font-mono">Two-Stage Neural Recommendation System</p>
            </div>
          </div>

          {/* Вкладки */}
          <div className="flex items-center bg-zinc-900/90 border border-zinc-800 p-1 rounded-xl">
            <button
              onClick={() => setViewMode('feed')}
              className={`px-4 py-1.5 rounded-lg text-xs font-mono font-bold flex items-center gap-2 transition-all ${
                viewMode === 'feed'
                  ? 'bg-amber-500 text-black shadow-md shadow-amber-500/20'
                  : 'text-zinc-400 hover:text-white'
              }`}
            >
              <Film className="w-3.5 h-3.5" />
              Кинолента
            </button>
            <button
              onClick={() => setViewMode('galaxy')}
              className={`px-4 py-1.5 rounded-lg text-xs font-mono font-bold flex items-center gap-2 transition-all ${
                viewMode === 'galaxy'
                  ? 'bg-amber-500 text-black shadow-md shadow-amber-500/20'
                  : 'text-zinc-400 hover:text-white'
              }`}
            >
              <Orbit className="w-3.5 h-3.5" />
              3D Галактика (WebGL)
            </button>
          </div>

          {/* Статусы */}
          <div className="flex items-center gap-2 font-mono text-xs">
            <span className="flex items-center gap-1.5 px-3 py-1 rounded-md bg-emerald-950/40 border border-emerald-500/30 text-emerald-400">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
              QDRANT 25k: UP
            </span>
            <span className="flex items-center gap-1.5 px-3 py-1 rounded-md bg-amber-950/40 border border-amber-500/30 text-amber-400">
              <Database className="w-3.5 h-3.5" />
              REDIS: ACTIVE
            </span>
          </div>
        </div>
      </header>

      {/* Основной контент */}
      <main className="max-w-7xl mx-auto px-6 py-8 flex-1 w-full space-y-8">
        
        {/* Контроллеры */}
        <section className="bg-[#0f0f14] border border-zinc-800/90 rounded-2xl p-6 shadow-2xl relative overflow-hidden">
          <div className="absolute top-0 left-0 w-1 h-full bg-amber-500"></div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 items-center">
            <div>
              <label className="text-[11px] font-bold text-zinc-400 uppercase tracking-wider block mb-2 font-mono">
                [USER_ID] Профиль вкуса
              </label>
              <div className="flex gap-1.5">
                {[42, 1, 100, 500].map(id => (
                  <button
                    key={id}
                    onClick={() => setUserId(id)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-mono font-bold transition-all ${
                      userId === id 
                        ? 'bg-amber-500 text-black shadow-md shadow-amber-500/30 ring-1 ring-amber-300' 
                        : 'bg-zinc-900 text-zinc-400 border border-zinc-800 hover:border-zinc-700 hover:text-white'
                    }`}
                  >
                    #{id}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <div className="flex justify-between items-center mb-2 font-mono text-xs">
                <span className="text-zinc-400 font-bold uppercase flex items-center gap-1">
                  <Sliders className="w-3.5 h-3.5 text-amber-500" />
                  MMR Разнообразие
                </span>
                <span className="text-amber-400 font-bold">{diversity}</span>
              </div>
              <input 
                type="range" min="0.0" max="1.0" step="0.1"
                value={diversity}
                onChange={e => setDiversity(parseFloat(e.target.value))}
                className="w-full h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-amber-500"
              />
            </div>

            <div>
              <div className="flex justify-between items-center mb-2 font-mono text-xs">
                <span className="text-zinc-400 font-bold uppercase flex items-center gap-1">
                  <Dice5 className="w-3.5 h-3.5 text-rose-500" />
                  Бандит (Разведка)
                </span>
                <span className="text-rose-400 font-bold">{Math.round(exploration * 100)}%</span>
              </div>
              <input 
                type="range" min="0.0" max="0.5" step="0.05"
                value={exploration}
                onChange={e => setExploration(parseFloat(e.target.value))}
                className="w-full h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-rose-500"
              />
            </div>

            <div>
              <button
                onClick={fetchRecommendations}
                disabled={loading}
                className="w-full py-2.5 px-4 bg-zinc-100 hover:bg-white text-black font-mono text-xs uppercase font-bold tracking-wider rounded-xl shadow-lg transition-all active:scale-95 disabled:opacity-50 flex items-center justify-center gap-2"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
                {loading ? 'Ранжирование...' : 'Синхронизировать'}
              </button>
            </div>
          </div>
        </section>

        {/* Задержки */}
        {data && (
          <div className="bg-[#0c0c10] border border-zinc-800/80 rounded-xl px-5 py-3 flex flex-wrap items-center justify-between gap-4 font-mono text-xs">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-emerald-400" />
              <span className="text-zinc-400">LATENCY E2E:</span>
              <span className="font-bold text-emerald-400 text-sm">{data.total_latency_ms} ms</span>
            </div>
            <div className="flex items-center gap-6 text-zinc-400 text-[11px]">
              <span>STAGE_1 (QDRANT): <strong className="text-zinc-200">{data.stage1_retrieval_ms}ms</strong></span>
              <span>STAGE_2 (CATBOOST): <strong className="text-zinc-200">{data.stage2_ranking_ms}ms</strong></span>
              <span>КАНДИДАТОВ: <strong className="text-zinc-200">{data.candidates_retrieved}</strong></span>
              <span>РАЗВЕДКА: <strong className="text-rose-400">{data.exploration_items_count} шт.</strong></span>
            </div>
          </div>
        )}

        {/* Вывод контента */}
        {viewMode === 'galaxy' ? (
          <section className="space-y-4">
            <div className="flex justify-between items-center">
              <div>
                <h2 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
                  <Orbit className="w-5 h-5 text-amber-500" />
                  3D Пространство скрытых эмбеддингов (Two-Tower Latent Space)
                </h2>
                <p className="text-xs text-zinc-400 mt-0.5">Вращай трехмерное облако фильмов и изучай кластеры вкусов</p>
              </div>
            </div>
            <Galaxy3D 
              recommendations={data?.recommendations || []} 
              userId={userId} 
              onSelectMovie={(movie) => setSelectedMovie(movie)}
            />
          </section>
        ) : (
          <section>
            <div className="flex justify-between items-center mb-6">
              <div>
                <h2 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
                  Рекомендательная выдача // Топ-10
                </h2>
                <p className="text-xs text-zinc-400 mt-0.5">Официальные постеры • Кликни на карточку для открытия инспектора</p>
              </div>
              <span className="text-xs font-mono px-2.5 py-1 rounded bg-zinc-900 border border-zinc-800 text-zinc-400">
                Ранжирование: CatBoost YetiRank
              </span>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-6">
              {data?.recommendations.map((movie) => {
                const isLiked = likedMovies.has(movie.movie_id);

                return (
                  <div
                    key={movie.movie_id}
                    onClick={() => setSelectedMovie(movie)}
                    className={`group relative rounded-xl overflow-hidden flex flex-col justify-between aspect-[2/3] p-4 cursor-pointer border transition-all duration-300 shadow-xl ${
                      movie.is_exploration 
                        ? 'border-rose-500/70 shadow-rose-950/40 hover:border-rose-400' 
                        : 'border-zinc-800/80 hover:border-amber-500/60 hover:shadow-2xl'
                    }`}
                  >
                    <MoviePoster movie={movie} />

                    <div className="flex justify-between items-start z-10">
                      <span className="px-2 py-0.5 rounded bg-black/70 backdrop-blur-md text-zinc-300 text-[11px] font-mono font-bold border border-white/10">
                        #{movie.rank}
                      </span>
                      {movie.is_exploration && (
                        <span className="px-2 py-0.5 rounded bg-rose-500 text-white text-[9px] font-mono font-black uppercase tracking-wider flex items-center gap-1 shadow-md shadow-rose-500/30">
                          <Dice5 className="w-2.5 h-2.5" />
                          РАЗВЕДКА
                        </span>
                      )}
                    </div>

                    <div className="z-10 mt-auto mb-2">
                      <span className="text-[10px] font-mono uppercase tracking-widest text-amber-400/90 block mb-0.5">
                        {movie.genres[0]}
                      </span>
                      {/* НОРМАЛИЗОВАННОЕ НАЗВАНИЕ */}
                      <h3 className="font-bold text-white text-base leading-snug line-clamp-2 group-hover:text-amber-300 transition-colors drop-shadow-md">
                        {normalizeTitle(movie.title)}
                      </h3>
                      {movie.year && (
                        <span className="text-xs font-mono text-zinc-400 mt-0.5 block">
                          {movie.year} г.
                        </span>
                      )}
                    </div>

                    <div className="pt-2.5 border-t border-white/10 flex items-center justify-between z-10 -mx-4 -mb-4 p-3 bg-black/60 backdrop-blur-md">
                      <div>
                        <div className="text-[9px] uppercase font-mono text-zinc-400">MATCH</div>
                        <div className="text-sm font-mono font-black text-amber-400">
                          {Math.round(movie.confidence_score * 100)}%
                        </div>
                      </div>

                      <button
                        onClick={(e) => handleLike(e, movie)}
                        className={`p-2 rounded-lg border transition-all active:scale-80 ${
                          isLiked 
                            ? 'bg-rose-500 text-white border-rose-400 shadow-lg shadow-rose-500/40' 
                            : 'bg-black/60 text-zinc-400 border-white/10 hover:text-rose-400 hover:border-rose-500/50'
                        }`}
                        title="Лайкнуть и сместить вектор в Redis"
                      >
                        <Heart className={`w-3.5 h-3.5 ${isLiked ? 'fill-current' : ''}`} />
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>
        )}

        <section className="bg-black border border-zinc-800/80 rounded-xl p-4 font-mono text-xs">
          <div className="flex items-center gap-2 text-zinc-400 mb-2.5 pb-2 border-b border-zinc-800 text-[11px] font-bold uppercase">
            <Terminal className="w-3.5 h-3.5 text-amber-500" />
            <span>Telemetry Event Stream (Redis & Engine Logs)</span>
          </div>
          <div className="space-y-1 text-zinc-400 text-[11px] max-h-24 overflow-y-auto">
            {logs.map((log, i) => (
              <div key={i} className={i === 0 ? "text-amber-400" : ""}>
                {log}
              </div>
            ))}
          </div>
        </section>
      </main>

      {/* Инспектор решения */}
      <AnimatePresence>
        {selectedMovie && (
          <div className="fixed inset-0 z-50 flex justify-end bg-black/75 backdrop-blur-sm">
            <motion.div
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={{ type: 'spring', damping: 30, stiffness: 300 }}
              className="w-full max-w-md bg-[#0e0e13] border-l border-zinc-800 h-full p-6 flex flex-col justify-between overflow-y-auto font-sans"
            >
              <div>
                <div className="flex justify-between items-center mb-6">
                  <span className="font-mono text-xs text-amber-500 font-bold uppercase">
                    Инспектор ML-ранжирования
                  </span>
                  <button 
                    onClick={() => setSelectedMovie(null)}
                    className="p-1.5 rounded-lg bg-zinc-900 border border-zinc-800 text-zinc-400 hover:text-white"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>

                {/* НОРМАЛИЗОВАННОЕ НАЗВАНИЕ В ИНСПЕКТОРЕ */}
                <h2 className="text-2xl font-bold text-white mb-1 leading-tight">
                  {normalizeTitle(selectedMovie.title)}
                </h2>
                <span className="font-mono text-sm text-zinc-400 block mb-6">
                  Релиз: {selectedMovie.year || 'Н/Д'} г. • ID #{selectedMovie.movie_id}
                </span>

                <div className="flex flex-wrap gap-1.5 mb-8">
                  {selectedMovie.genres.map(g => (
                    <span key={g} className="px-2.5 py-1 rounded bg-zinc-900 border border-zinc-800 text-zinc-300 text-xs font-mono">
                      {g}
                    </span>
                  ))}
                </div>

                <div className="space-y-4 bg-zinc-900/60 p-5 rounded-xl border border-zinc-800/80 mb-6 font-mono">
                  <h4 className="text-xs uppercase font-bold text-zinc-400 flex items-center gap-1.5">
                    <BarChart3 className="w-4 h-4 text-amber-500" />
                    Вклад алгоритмов в решение:
                  </h4>

                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-zinc-400">Уверенность CatBoost:</span>
                      <span className="text-amber-400 font-bold">{Math.round(selectedMovie.confidence_score * 100)}%</span>
                    </div>
                    <div className="w-full bg-zinc-800 h-2 rounded-full overflow-hidden">
                      <div className="bg-amber-500 h-full rounded-full" style={{ width: `${selectedMovie.confidence_score * 100}%` }}></div>
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-zinc-400">Косинус Two-Tower (Stage 1):</span>
                      <span className="text-zinc-200">{selectedMovie.retrieval_score}</span>
                    </div>
                    <div className="w-full bg-zinc-800 h-2 rounded-full overflow-hidden">
                      <div className="bg-zinc-400 h-full rounded-full" style={{ width: `${Math.max(0, selectedMovie.retrieval_score * 100)}%` }}></div>
                    </div>
                  </div>
                </div>
              </div>

              <button
                onClick={() => setSelectedMovie(null)}
                className="w-full py-2.5 bg-zinc-800 hover:bg-zinc-700 text-white font-mono text-xs uppercase font-bold rounded-xl transition-colors"
              >
                Закрыть инспектор
              </button>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}