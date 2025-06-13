import React, { useState } from "react";
import axios from "axios";

export default function UploadForm() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("");
  const [clipResults, setClipResults] = useState([]);
  const [selectedClip, setSelectedClip] = useState(null);
  const [showModal, setShowModal] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setClipResults([]);
    setStatus("");
  };

  const handleUpload = async () => {
    if (!file) {
      setStatus("Please select a file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setStatus("⏳ Uploading and processing...");
      const res = await axios.post("/api/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      if (res.data?.filename) {
        const processRes = await axios.post("/api/process", {
          filename: res.data.filename,
        });

        if (processRes.data?.highlights) {
          setClipResults(processRes.data.highlights);
          setStatus(`✅ Processed ${processRes.data.highlights.length} clips successfully.`);
        } else {
          setStatus("⚠️ No highlights returned.");
        }
      } else {
        setStatus("⚠️ Upload succeeded, but no filename returned.");
      }
    } catch (err) {
      console.error(err);
      setStatus("❌ Upload failed. Please check the server.");
    }
  };

  return (
    <div className="max-w-5xl mx-auto mt-8 p-6 bg-white rounded shadow">
      <h2 className="text-xl font-semibold mb-4">Upload a Podcast File</h2>
      <input type="file" accept="audio/*,video/*" onChange={handleFileChange} className="mb-4" />
      <button onClick={handleUpload} className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
        Upload
      </button>
      <p className="mt-2 text-sm text-gray-600">{status}</p>

      {clipResults.length > 0 && (
        <div className="mt-8 overflow-x-auto">
          <h3 className="text-lg font-bold mb-2">Clip Results:</h3>
          <table className="min-w-full bg-white border">
            <thead>
              <tr>
                <th className="py-2 px-4 border">Clip No.</th>
                <th className="py-2 px-4 border">View</th>
                <th className="py-2 px-4 border">LLM Score</th>
                <th className="py-2 px-4 border">Virality Score</th>
                <th className="py-2 px-4 border">Viral?</th>
                <th className="py-2 px-4 border">Download</th>
              </tr>
            </thead>
            <tbody>
              {clipResults.map((clip, idx) => (
                <tr key={idx} className="text-center">
                  <td className="py-2 px-4 border">Clip_{idx + 1}</td>
                  <td className="py-2 px-4 border">
                    <button
                      className="text-blue-600 underline"
                      onClick={() => {
                        setSelectedClip(`/static/${clip.clip_path}`);
                        setShowModal(true);
                      }}
                    >
                      View
                    </button>
                  </td>
                  <td className="py-2 px-4 border">{clip.llm_score}</td>
                  <td className="py-2 px-4 border">{clip.virality_score}</td>
                  <td className="py-2 px-4 border">
                    {clip.label === 1 ? (
                      <span className="text-green-600 font-semibold">Viral 🚀</span>
                    ) : (
                      <span className="text-gray-600">Not Viral ❄️</span>
                    )}
                  </td>
                  <td className="py-2 px-4 border">
                    <a
                      href={`/static/${clip.clip_path}`}
                      download
                      className="text-blue-600 underline"
                    >
                      Download
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Modal */}
      {showModal && selectedClip && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-70 z-50">
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full">
            <h4 className="text-lg font-semibold mb-2">Clip Preview</h4>
            <video controls src={selectedClip} className="w-full h-auto rounded mb-4" />
            <div className="text-right">
              <button
                className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
                onClick={() => {
                  setShowModal(false);
                  setSelectedClip(null);
                }}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
