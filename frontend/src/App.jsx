import UploadForm from "./components/UploadForm";
import "./index.css";

function App() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-100 to-white text-gray-900">
      <header className="bg-gradient-to-r from-indigo-600 to-blue-600 text-white py-6 text-center shadow-lg">
        <h1 className="text-3xl font-bold tracking-tight">🎬 Podcast-to-Reels Generator</h1>
        <p className="text-sm font-light mt-1">AI-powered smart clip extractor</p>
      </header>
      <main className="p-6 max-w-5xl mx-auto">
        <UploadForm />
      </main>
    </div>
  );
}

export default App;
