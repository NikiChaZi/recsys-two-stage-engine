import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

const GENRE_COLORS = {
  'Sci-Fi': 0x06b6d4,
  'Action': 0xf59e0b,
  'Horror': 0xef4444,
  'Comedy': 0xeab308,
  'Drama': 0x3b82f6,
  'Western': 0xd97706,
  'Fantasy': 0x10b981,
  'War': 0x71717a,
  'default': 0x8b5cf6
};

export default function Galaxy3D({ recommendations = [], userId = 42, onSelectMovie }) {
  const containerRef = useRef(null);
  const [hoveredMovie, setHoveredMovie] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });

  useEffect(() => {
    if (!containerRef.current) return;

    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x08080a, 0.002);

    const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 2000);
    camera.position.set(0, 40, 240);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    containerRef.current.innerHTML = '';
    containerRef.current.appendChild(renderer.domElement);

    // 1. Фоновые звезды
    const starCount = 2000;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(starCount * 3);
    const colors = new Float32Array(starCount * 3);
    const genres = Object.keys(GENRE_COLORS).filter(g => g !== 'default');
    
    for (let i = 0; i < starCount; i++) {
      const genreIndex = i % genres.length;
      const color = new THREE.Color(GENRE_COLORS[genres[genreIndex]]);
      const angle = (genreIndex / genres.length) * Math.PI * 2;
      const radius = 60 + Math.random() * 120;
      const spread = 35;

      positions[i * 3] = Math.cos(angle) * radius + (Math.random() - 0.5) * spread;
      positions[i * 3 + 1] = (Math.random() - 0.5) * spread * 1.5;
      positions[i * 3 + 2] = Math.sin(angle) * radius + (Math.random() - 0.5) * spread;

      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const starMaterial = new THREE.PointsMaterial({ size: 2.2, vertexColors: true, transparent: true, opacity: 0.6 });
    const starField = new THREE.Points(geometry, starMaterial);
    scene.add(starField);

    // 2. Вектор юзера
    const userGeo = new THREE.SphereGeometry(4.5, 32, 32);
    const userMat = new THREE.MeshBasicMaterial({ color: 0x10b981, wireframe: true });
    const userMesh = new THREE.Mesh(userGeo, userMat);
    scene.add(userMesh);

    // 3. Top-10 Рекомендаций
    const topMeshes = [];
    recommendations.forEach((rec, idx) => {
      const angle = (idx / recommendations.length) * Math.PI * 2;
      const dist = 75 + (rec.retrieval_score || 0.5) * 35;
      const x = Math.cos(angle) * dist;
      const y = (Math.sin(idx * 2) * 22);
      const z = Math.sin(angle) * dist;

      const sphereGeo = new THREE.SphereGeometry(rec.is_exploration ? 4.5 : 3.5, 32, 32);
      const sphereMat = new THREE.MeshBasicMaterial({ 
        color: rec.is_exploration ? 0xf43f5e : 0xf59e0b 
      });
      const mesh = new THREE.Mesh(sphereGeo, sphereMat);
      mesh.position.set(x, y, z);
      mesh.userData = rec;
      scene.add(mesh);
      topMeshes.push(mesh);

      // Луч
      const lineMat = new THREE.LineBasicMaterial({ 
        color: rec.is_exploration ? 0xf43f5e : 0xf59e0b, 
        transparent: true, 
        opacity: 0.35 
      });
      const lineGeo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3(x, y, z)
      ]);
      scene.add(new THREE.Line(lineGeo, lineMat));
    });

    // 4. Raycaster для кликов и наведений
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    const onPointerMove = (e) => {
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(topMeshes);

      if (intersects.length > 0) {
        const hit = intersects[0].object.userData;
        setHoveredMovie(hit);
        setTooltipPos({ x: e.clientX - rect.left + 15, y: e.clientY - rect.top - 20 });
        renderer.domElement.style.cursor = 'pointer';
      } else {
        setHoveredMovie(null);
        renderer.domElement.style.cursor = isDragging ? 'grabbing' : 'grab';
      }
    };

    const onClick = (e) => {
      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(topMeshes);
      if (intersects.length > 0) {
        const hit = intersects[0].object.userData;
        if (onSelectMovie) onSelectMovie(hit);
      }
    };

    // 5. Вращение и Блокировка прокрутки страницы
    let isDragging = false;
    let prevMouse = { x: 0, y: 0 };
    let rotation = { x: 0, y: 0 };

    const onMouseDown = (e) => {
      isDragging = true;
      prevMouse = { x: e.clientX, y: e.clientY };
    };

    const onMouseMoveWindow = (e) => {
      if (!isDragging) return;
      rotation.y += (e.clientX - prevMouse.x) * 0.005;
      rotation.x += (e.clientY - prevMouse.y) * 0.005;
      prevMouse = { x: e.clientX, y: e.clientY };
    };

    const onMouseUpWindow = () => { isDragging = false; };

    // ВАЖНО: e.preventDefault() намертво блокирует прокрутку страницы!
    const onWheel = (e) => {
      e.preventDefault();
      camera.position.z = Math.max(70, Math.min(450, camera.position.z + e.deltaY * 0.25));
    };

    const dom = renderer.domElement;
    dom.addEventListener('mousedown', onMouseDown);
    dom.addEventListener('mousemove', onPointerMove);
    dom.addEventListener('click', onClick);
    dom.addEventListener('wheel', onWheel, { passive: false }); // <-- passive: false чтобы работал preventDefault
    window.addEventListener('mousemove', onMouseMoveWindow);
    window.addEventListener('mouseup', onMouseUpWindow);

    let reqId;
    let clock = new THREE.Clock();

    const animate = () => {
      reqId = requestAnimationFrame(animate);
      const elapsed = clock.getElapsedTime();

      starField.rotation.y = rotation.y + elapsed * 0.02;
      starField.rotation.x = rotation.x;
      userMesh.rotation.y += 0.02;
      userMesh.scale.setScalar(1 + Math.sin(elapsed * 3) * 0.15);

      scene.rotation.y = rotation.y;
      scene.rotation.x = rotation.x;

      renderer.render(scene, camera);
    };

    animate();

    return () => {
      cancelAnimationFrame(reqId);
      dom.removeEventListener('mousedown', onMouseDown);
      dom.removeEventListener('mousemove', onPointerMove);
      dom.removeEventListener('click', onClick);
      dom.removeEventListener('wheel', onWheel);
      window.removeEventListener('mousemove', onMouseMoveWindow);
      window.removeEventListener('mouseup', onMouseUpWindow);
      renderer.dispose();
    };
  }, [recommendations]);

  return (
    <div className="relative w-full h-[650px] rounded-2xl overflow-hidden border border-zinc-800 bg-[#060608] shadow-2xl select-none">
      <div ref={containerRef} className="w-full h-full cursor-grab active:cursor-grabbing" />
      
      {/* Подсказка */}
      <div className="absolute top-4 left-4 z-10 bg-black/75 backdrop-blur-md px-4 py-2.5 rounded-xl border border-white/10 font-mono text-[11px] text-zinc-300 pointer-events-none space-y-1">
        <span className="text-emerald-400 font-bold block">● ВЕКТОР USER #{userId} (ЦЕНТР)</span>
        <span className="text-amber-400 block">◆ ТОП-10 РЕКОМЕНДАЦИЙ (ПЛАНЕТЫ)</span>
        <span className="text-zinc-400 block pt-1 border-t border-zinc-800">
          Наведи на планету для инфы • Кликни для открытия CatBoost
        </span>
      </div>

      {/* Всплывающий тултип над планетой с МИНИ-ПОСТЕРОМ */}
      {hoveredMovie && (
        <div 
          className="absolute z-20 pointer-events-none bg-[#0e0e13]/95 border border-amber-500/60 p-3 rounded-xl shadow-2xl backdrop-blur-md max-w-xs transition-all flex gap-3 items-center"
          style={{ left: tooltipPos.x, top: tooltipPos.y }}
        >
          {localStorage.getItem(`poster_${hoveredMovie.movie_id}`) && (
            <img 
              src={localStorage.getItem(`poster_${hoveredMovie.movie_id}`)} 
              alt="" 
              className="w-12 h-16 object-cover rounded-md flex-shrink-0 border border-white/10 shadow-md"
            />
          )}
          <div>
            <div className="flex justify-between items-center mb-1">
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 font-bold">
                #{hoveredMovie.rank}
              </span>
              <span className="text-[10px] font-mono font-bold text-emerald-400">
                {Math.round(hoveredMovie.confidence_score * 100)}%
              </span>
            </div>
            <h4 className="font-bold text-white text-xs leading-snug">{hoveredMovie.title}</h4>
            <span className="text-[10px] font-mono text-zinc-400 block mt-0.5">{hoveredMovie.genres.join(', ')}</span>
            <span className="text-[9px] font-mono text-amber-400/80 mt-1 block">Кликни для открытия инспектора →</span>
          </div>
        </div>
      )}

      <div className="absolute bottom-4 right-4 z-10 bg-black/60 backdrop-blur-md px-3.5 py-2 rounded-xl border border-white/10 font-mono text-[11px] text-zinc-400 pointer-events-none">
        Движок: WebGL 3D • Raycasting Enabled
      </div>
    </div>
  );
}