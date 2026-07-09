import { useState } from "react";

export default function AnnotationPicker() {
  const [data, setData] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [results, setResults] = useState([]);

  const handleFileUpload = (e) => {
    const reader = new FileReader();
    reader.onload = () => setData(JSON.parse(reader.result));
    reader.readAsText(e.target.files[0]);
  };

  const handlePick = (annotationIndex, shortAnswerIndex = null) => {
    const ann = data[currentIndex].annotations[annotationIndex];
    const picked = {
      question: data[currentIndex].question,
      gt_answer: ann.short_answers.length > 0 ? ann.short_answers[shortAnswerIndex] : ann.yes_no_answer,
      picked_annotation: [annotationIndex, ann.yes_no_answer=="NONE" ? "short_answer" : "yes_no", shortAnswerIndex]
    };
    const updated = [...results];
    updated[currentIndex] = picked;
    setResults(updated);
    setCurrentIndex((prev) => Math.min(prev + 1, data.length - 1));
  };

  const downloadResults = () => {
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'results.json';
    a.click();
  };

  if (data.length === 0) {
    return <input type="file" onChange={handleFileUpload} className="p-4" />;
  }

  const current = data[currentIndex];
  const currentResult = results[currentIndex];

  return (
    <div className="p-4 space-y-4">
      <div className="flex justify-between">
        <h2 className="text-xl font-bold">Question {currentIndex + 1}/{data.length}</h2>
        <button onClick={downloadResults} className="bg-blue-500 text-white px-2 py-1 rounded">Download Results</button>
      </div>
      <p className="text-lg">{current.question}</p>
      <div className="space-y-2">
        {current.annotations.map((ann, idx) => (
          <div key={idx} className={`border rounded p-2 ${currentResult?.picked_annotation?.[0] === idx ? 'border-green-500 border-4' : ''}`}>
            <div className="space-y-2">
              {ann.short_answers.length > 0 ? (
                <div>
                  <p className="font-semibold">Short Answers:</p>
                  {ann.short_answers.map((sa, saIdx) => (
                    <button
                      key={saIdx}
                      onClick={() => handlePick(idx, saIdx)}
                      className="bg-green-500 text-white px-2 py-1 rounded m-1"
                    >
                      {sa}
                    </button>
                  ))}
                </div>
              ) : (
                <div>
                  <p className="font-semibold">Yes/No Answer:</p>
                  <button onClick={() => handlePick(idx)} className="bg-green-500 text-white px-2 py-1 rounded">{ann.yes_no_answer}</button>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
      <div className="flex justify-between mt-4">
        <button onClick={() => setCurrentIndex((prev) => Math.max(prev - 1, 0))} className="bg-gray-500 text-white px-2 py-1 rounded">Previous</button>
        <button onClick={() => setCurrentIndex((prev) => Math.min(prev + 1, data.length - 1))} className="bg-gray-500 text-white px-2 py-1 rounded">Next</button>
      </div>
    </div>
  );
}
