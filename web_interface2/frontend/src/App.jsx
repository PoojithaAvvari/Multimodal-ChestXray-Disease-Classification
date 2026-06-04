import { useState } from 'react';
import axios from 'axios';
import { Upload, FileText, Activity, AlertTriangle, CheckCircle2, X, ShieldAlert } from 'lucide-react';
import clsx from 'clsx';
import { twMerge } from 'tailwind-merge';

function cx(...inputs) {
  return twMerge(clsx(inputs));
}

function App() {
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [reportFile, setReportFile] = useState(null);
  const [reportPreview, setReportPreview] = useState(null);
  const [reportText, setReportText] = useState('');
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [fakeAlert, setFakeAlert] = useState(null);
  const [rejectedAlert, setRejectedAlert] = useState(null);
  const [results, setResults] = useState(null);
  const [diseaseFindings, setDiseaseFindings] = useState(null);
  const [cxrConfidence, setCxrConfidence] = useState(null);
  const [analysisMode, setAnalysisMode] = useState(null); // 'dual_pipeline', 'image_only', 'report_only'
  const [pipelineStatus, setPipelineStatus] = useState(null); // Track pipeline success/failures

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      setImagePreview(URL.createObjectURL(file));
      resetState();
    }
  };

  const handleReportChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setReportFile(file);
      setReportPreview(URL.createObjectURL(file));
      setReportText('');
      resetState();
    }
  };

  const handleReportTextChange = (e) => {
    setReportText(e.target.value);
    setReportFile(null);
    setReportPreview(null);
    resetState();
  };

  const resetState = () => {
    setFakeAlert(null);
    setRejectedAlert(null);
    setResults(null);
    setDiseaseFindings(null);
    setCxrConfidence(null);
    setAnalysisMode(null);
    setPipelineStatus(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!imageFile && !reportFile && !reportText) {
      setError("Please provide an X-Ray image or a clinical report (image or text).");
      return;
    }

    setLoading(true);
    resetState();

    const formData = new FormData();
    if (imageFile) formData.append('image', imageFile);
    if (reportFile) formData.append('report_image', reportFile);
    if (reportText) formData.append('report_text', reportText);

    try {
      const response = await axios.post('http://127.0.0.1:8000/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      const data = response.data;

      if (data.status === 'fake') {
        setFakeAlert(data);
      } else if (data.status === 'rejected') {
        // Both pipelines failed
        setPipelineStatus(data);
        setRejectedAlert({
           isGeneralRejection: true,
           imageError: data.image_pipeline?.error || 'No input',
           reportError: data.report_pipeline?.error || 'No input',
           reason: data.reason
        });
      } else if (data.status === 'success') {
        // Handle different pipeline modes
        setAnalysisMode(data.mode || data.source);
        setPipelineStatus({
          image: data.image_pipeline,
          report: data.report_pipeline
        });
        
        if (data.message) {
          console.log("Pipeline partial success:", data.message);
        }
        
        setDiseaseFindings(data.disease_findings || []);
        setCxrConfidence(data.image_pipeline?.cxr_confidence || data.chest_xray_confidence);
      } else if (data.status === 'real') {
        // Old multimodal model response (report-only path)
        setResults(data.results);
      }
    } catch (err) {
      console.error(err);
      setError("An error occurred while analyzing the data. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const getLevelColor = (level) => {
    switch (level) {
      case 'Definite': return { bg: 'bg-red-500/20', border: 'border-red-400', text: 'text-red-300', bar: 'bg-gradient-to-r from-red-500 to-rose-400 shadow-[0_0_10px_rgba(239,68,68,0.8)]', badge: 'bg-red-500/20 text-red-400 border-red-500/30' };
      case 'Probable': return { bg: 'bg-orange-500/20', border: 'border-orange-400', text: 'text-orange-300', bar: 'bg-gradient-to-r from-orange-500 to-amber-400 shadow-[0_0_10px_rgba(249,115,22,0.8)]', badge: 'bg-orange-500/20 text-orange-400 border-orange-500/30' };
      case 'Possible': return { bg: 'bg-yellow-500/15', border: 'border-yellow-400/60', text: 'text-yellow-300', bar: 'bg-gradient-to-r from-yellow-500 to-amber-300 shadow-[0_0_10px_rgba(234,179,8,0.6)]', badge: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30' };
      case 'Unlikely': return { bg: 'bg-slate-800/30', border: 'border-white/5', text: 'text-slate-300', bar: 'bg-slate-600', badge: 'bg-slate-700/50 text-slate-400 border-slate-600/50' };
      default: return { bg: 'bg-slate-800/30', border: 'border-white/5', text: 'text-slate-300', bar: 'bg-slate-600', badge: 'bg-slate-700/50 text-slate-400 border-slate-600/50' };
    }
  };

  return (
    <div className="min-h-screen mesh-bg relative flex flex-col items-center py-12 px-4 sm:px-6 lg:px-8">
      {/* Abstract Background Blobs */}
      <div className="blob bg-indigo-600 w-96 h-96 top-0 left-0"></div>
      <div className="blob bg-blue-600 w-96 h-96 bottom-0 right-0"></div>

      {/* Main Container */}
      <div className="max-w-6xl w-full space-y-8 z-10 glass p-8 rounded-3xl">
        <header className="text-center space-y-2">
          <div className="inline-flex items-center justify-center p-3 bg-blue-500/10 rounded-2xl mb-4 border border-blue-500/20 shadow-[0_0_15px_rgba(59,130,246,0.5)]">
            <Activity className="w-8 h-8 text-blue-400" />
          </div>
          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-white mb-2">
            Multi-Modal <span className="text-gradient">Chest X-Ray</span> Classifier
          </h1>
          <p className="text-slate-400 max-w-2xl mx-auto text-lg">
            Upload a chest X-ray and clinical report. Our dual-model system will automatically verify authenticity before generating multimodal disease classifications.
          </p>
        </header>

        {error && (
          <div className="bg-red-500/10 border border-red-500/50 text-red-200 p-4 rounded-xl flex items-center gap-3">
            <AlertTriangle className="w-5 h-5 flex-shrink-0" />
            <p>{error}</p>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Left Column: Image Upload */}
          <div className="glass-card p-6 flex flex-col">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2 text-blue-300">
              <Upload className="w-5 h-5" />
              1. Upload Chest X-Ray
            </h2>
            <div className="flex-1 min-h-[250px] border-2 border-dashed border-slate-600 hover:border-blue-400/50 rounded-xl transition-all duration-300 overflow-hidden relative group bg-slate-800/20">
              <input 
                key={imagePreview || 'empty-img'}
                type="file" 
                accept="image/*" 
                onChange={handleImageChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20"
              />
              {imagePreview ? (
                <>
                  <div className="absolute inset-0 z-10 p-2">
                    <img src={imagePreview} alt="Preview" className="w-full h-full object-cover rounded-lg" />
                    <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center pointer-events-none rounded-lg">
                      <p className="text-white font-medium">Click to change image</p>
                    </div>
                  </div>
                  <button 
                    onClick={(e) => { e.preventDefault(); e.stopPropagation(); setImageFile(null); setImagePreview(null); }}
                    className="absolute top-4 right-4 z-30 p-1.5 bg-red-500/90 rounded-full text-white hover:bg-red-600 transition-all shadow-lg hover:scale-110"
                    title="Remove Image"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </>
              ) : (
                <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-400 pointer-events-none z-10 p-6 text-center">
                  <Upload className="w-12 h-12 mb-3 text-slate-500 group-hover:text-blue-400 transition-colors" />
                  <p className="text-sm font-medium">Drag & drop or click to select</p>
                  <p className="text-xs mt-1 text-slate-500">Supports PNG, JPG, JPEG</p>
                </div>
              )}
            </div>
          </div>

          {/* Right Column: Report Upload & Text */}
          <div className="glass-card p-6 flex flex-col">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2 text-blue-300">
              <FileText className="w-5 h-5" />
              2. Clinical Report
            </h2>
            
            <div className="flex flex-col gap-4 flex-1">
              {/* Report Image Upload */}
              <div className="min-h-[140px] border-2 border-dashed border-slate-600 hover:border-blue-400/50 rounded-xl transition-all duration-300 overflow-hidden relative group bg-slate-800/20">
                <input 
                  key={reportPreview || 'empty-report'}
                  type="file" 
                  accept="image/*" 
                  onChange={handleReportChange}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20"
                />
                {reportPreview ? (
                  <>
                    <div className="absolute inset-0 z-10 p-2">
                      <img src={reportPreview} alt="Report Preview" className="w-full h-full object-cover rounded-lg opacity-80" />
                      <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center pointer-events-none rounded-lg">
                        <p className="text-white font-medium">Click to change image</p>
                      </div>
                    </div>
                    <button 
                      onClick={(e) => { e.preventDefault(); e.stopPropagation(); setReportFile(null); setReportPreview(null); }}
                      className="absolute top-4 right-4 z-30 p-1.5 bg-red-500/90 rounded-full text-white hover:bg-red-600 transition-all shadow-lg hover:scale-110"
                      title="Remove Report Image"
                    >
                      <X className="w-5 h-5" />
                    </button>
                  </>
                ) : (
                  <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-400 pointer-events-none z-10 p-4 text-center">
                    <Upload className="w-8 h-8 mb-2 text-slate-500 group-hover:text-blue-400 transition-colors" />
                    <p className="text-sm font-medium">Upload handwritten notes (PNG/JPG)</p>
                  </div>
                )}
              </div>

              {/* Report Text Input */}
              {/* <div className="flex flex-col gap-2">
                <p className="text-xs text-slate-400 font-medium px-1">Or enter clinical report text:</p>
                <textarea
                  value={reportText}
                  onChange={handleReportTextChange}
                  placeholder="Paste or type clinical notes, findings, diagnoses, patient history, etc..."
                  className="w-full h-[100px] bg-slate-900/50 border border-slate-600 rounded-xl px-4 py-3 text-slate-100 placeholder-slate-500 focus:border-blue-400 focus:outline-none focus:ring-1 focus:ring-blue-400/50 transition-all resize-none"
                />
              </div> */}
            </div>
          </div>
        </div>

        {/* Action Button */}
        <div className="flex justify-center mt-8">
          <button 
            onClick={handleAnalyze}
            disabled={loading || (!imageFile && !reportFile && !reportText)}
            className="group relative px-8 py-4 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white rounded-xl font-bold text-lg shadow-[0_0_20px_rgba(59,130,246,0.4)] disabled:opacity-50 disabled:cursor-not-allowed transition-all hover:scale-[1.02] flex items-center gap-3 overflow-hidden"
          >
            {loading ? (
              <>
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                Analyzing Multi-Modal Data...
              </>
            ) : (
              <>
                <Activity className="w-5 h-5" />
                Run AI Diagnostics
                <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300 pointer-events-none"></div>
              </>
            )}
          </button>
        </div>

        {/* Disease Findings Results — New Pipeline (BiomedCLIP + TorchXRayVision) */}
        {diseaseFindings && (
          <div className="mt-12 glass-card p-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {/* Pipeline Status Indicators */}
            {pipelineStatus && (pipelineStatus.image?.status === 'failed' || pipelineStatus.report?.status === 'failed') && (
              <div className="mb-6 p-4 bg-amber-500/10 border border-amber-500/30 rounded-lg">
                <div className="flex items-start gap-3">
                  <AlertTriangle className="w-5 h-5 text-amber-500 flex-shrink-0 mt-0.5" />
                  <div className="flex-1 text-sm text-amber-200">
                    {pipelineStatus.image?.status === 'failed' && pipelineStatus.report?.status === 'failed' && (
                      <div>
                        <p className="font-semibold mb-2">⚠️ Both Pipelines Encountered Issues:</p>
                        <p className="text-amber-300 mb-1">📸 X-Ray: {pipelineStatus.image?.error}</p>
                        <p className="text-amber-300">📄 Report: {pipelineStatus.report?.error}</p>
                      </div>
                    )}
                    {pipelineStatus.image?.status === 'failed' && pipelineStatus.report?.status === 'success' && (
                      <p>⚠️ X-Ray analysis failed ({pipelineStatus.image?.error}). Showing report analysis only.</p>
                    )}
                    {pipelineStatus.image?.status === 'success' && pipelineStatus.report?.status === 'failed' && (
                      <p>⚠️ Report analysis failed ({pipelineStatus.report?.error}). Showing X-ray analysis only.</p>
                    )}
                  </div>
                </div>
              </div>
            )}
            
            <div className="flex items-center justify-between mb-8 pb-4 border-b border-slate-700/50 flex-wrap gap-4">
              <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                <CheckCircle2 className="w-8 h-8 text-emerald-400" />
                Disease Detection Results
              </h2>
              <div className="flex items-center gap-3 flex-wrap">
                {analysisMode === 'dual_pipeline' && (
                  <>
                    <div className="px-4 py-1.5 bg-blue-500/10 border border-blue-500/20 text-blue-300 rounded-full text-sm font-medium flex items-center gap-2">
                      📸 X-Ray + Report
                    </div>
                    <div className="px-3 py-1 bg-slate-700/40 border border-slate-600 text-slate-300 rounded text-xs">
                      Confidence-Based Ensemble
                    </div>
                  </>
                )}
                {analysisMode === 'image_only' && (
                  <div className="px-4 py-1.5 bg-blue-500/10 border border-blue-500/20 text-blue-400 rounded-full text-sm font-medium flex items-center gap-2">
                    ✓ From X-Ray Image
                  </div>
                )}
                {analysisMode === 'report_only' && (
                  <div className="px-4 py-1.5 bg-purple-500/10 border border-purple-500/20 text-purple-400 rounded-full text-sm font-medium flex items-center gap-2">
                    ✓ From Clinical Report
                  </div>
                )}
                {cxrConfidence && !isNaN(cxrConfidence) && analysisMode !== 'report_only' && (
                  <div className="px-4 py-1.5 bg-blue-500/10 border border-blue-500/20 text-blue-400 rounded-full text-sm font-medium flex items-center gap-2">
                    Chest X-Ray: {(parseFloat(cxrConfidence) * 100).toFixed(1)}%
                  </div>
                )}
              </div>
            </div>

            {diseaseFindings.length === 0 ? (
              <div className="text-center py-12">
                <CheckCircle2 className="w-16 h-16 text-emerald-400 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-white mb-2">No Significant Abnormalities</h3>
                <p className="text-slate-400">Chest X-ray appears NORMAL. No pathologies detected above threshold.</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {diseaseFindings.map((finding, idx) => {
                  const colors = getLevelColor(finding.level);
                  const hasSources = finding.sources && finding.sources.length > 0;
                  
                  return (
                    <div key={idx} className={cx(
                      "p-5 rounded-xl border transition-all",
                      colors.bg, colors.border,
                      finding.confidence > 0.5 && `shadow-[0_0_20px_rgba(59,130,246,0.2)]`
                    )}>
                      <div className="flex justify-between items-start mb-3 gap-2">
                        <div className="flex-1">
                          <span className={cx("font-semibold text-lg block", colors.text)}>
                            {finding.disease.replace(/_/g, ' ')}
                          </span>
                          {hasSources && (
                            <div className="flex gap-1.5 mt-1.5">
                              {finding.sources.includes('image') && (
                                <span className="text-[10px] px-2 py-0.5 bg-blue-500/20 text-blue-300 rounded border border-blue-400/30">
                                  From X-Ray
                                </span>
                              )}
                              {finding.sources.includes('report') && (
                                <span className="text-[10px] px-2 py-0.5 bg-purple-500/20 text-purple-300 rounded border border-purple-400/30">
                                  From Report
                                </span>
                              )}
                            </div>
                          )}
                        </div>
                        <span className={cx("text-xs font-bold px-2.5 py-1 rounded-full border", colors.badge)}>
                          {finding.level}
                        </span>
                      </div>
                      
                      {/* Show individual confidences if ensemble (both must exist and be numbers) */}
                      {finding.image_confidence != null && finding.report_confidence != null && typeof finding.image_confidence === 'number' && typeof finding.report_confidence === 'number' && (
                        <div className="mb-3 text-xs space-y-1 bg-slate-900/30 p-2 rounded">
                          <div className="flex justify-between text-slate-400">
                            <span>Image:</span>
                            <span className="text-blue-300 font-mono">{!isNaN(finding.image_confidence) ? (finding.image_confidence * 100).toFixed(1) : 'N/A'}%</span>
                          </div>
                          <div className="flex justify-between text-slate-400">
                            <span>Report:</span>
                            <span className="text-purple-300 font-mono">{!isNaN(finding.report_confidence) ? (finding.report_confidence * 100).toFixed(1) : 'N/A'}%</span>
                          </div>
                        </div>
                      )}
                      
                      <div className="flex items-center gap-3">
                        <div className="flex-1 bg-slate-900 rounded-full h-2.5 overflow-hidden">
                          <div 
                            className={cx("h-2.5 rounded-full transition-all duration-1000 ease-out", colors.bar)} 
                            style={{ width: `${!isNaN(finding.confidence) ? finding.confidence * 100 : 0}%` }}
                          ></div>
                        </div>
                        <span className={cx("text-sm font-mono min-w-[50px] text-right", colors.text)}>
                          {!isNaN(finding.confidence) ? (parseFloat(finding.confidence) * 100).toFixed(1) : 'N/A'}%
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* <div className="mt-6 pt-4 border-t border-slate-700/30">
              <p className="text-xs text-slate-500 text-center">
                Powered by TorchXRayVision DenseNet-121 · 18 Pathology Detection · Validated by BiomedCLIP
              </p>
            </div> */}
          </div>
        )}

        {/* Old Model Results — kept for report-only flow */}
        {results && (
          <div className="mt-12 glass-card p-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="flex items-center justify-between mb-8 pb-4 border-b border-slate-700/50">
              <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                <CheckCircle2 className="w-8 h-8 text-emerald-400" />
                Diagnostic Analysis
              </h2>
              <div className="px-4 py-1.5 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 rounded-full text-sm font-medium flex items-center gap-2">
                Authentic Image Verified
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {results.slice().sort((a, b) => b.probability - a.probability).map((res, idx) => (
                <div key={idx} className={cx(
                  "p-5 rounded-xl border transition-all",
                  res.positive 
                    ? "bg-blue-500/20 border-blue-400 shadow-[0_0_20px_rgba(59,130,246,0.4)]" 
                    : "bg-slate-800/30 border-white/5"
                )}>
                  <div className="flex justify-between items-end mb-2">
                    <span className={cx(
                      "font-semibold text-lg",
                      res.positive ? "text-blue-300" : "text-slate-300"
                    )}>
                      {res.label}
                    </span>
                    <span className={cx(
                      "text-sm font-mono",
                      res.positive ? "text-blue-400" : "text-slate-500"
                    )}>
                      {(res.probability * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-slate-900 rounded-full h-2.5 overflow-hidden">
                    <div 
                      className={cx(
                        "h-2.5 rounded-full transition-all duration-1000 ease-out",
                        res.positive ? "bg-gradient-to-r from-blue-500 to-indigo-400 shadow-[0_0_10px_rgba(59,130,246,0.8)]" : "bg-slate-600"
                      )} 
                      style={{ width: `${res.probability * 100}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Medical Disclaimer */}
        <div className="mt-12 bg-amber-500/10 border border-amber-500/30 p-6 rounded-2xl flex flex-col sm:flex-row gap-4 items-center sm:items-start text-amber-200/80">
          <AlertTriangle className="w-8 h-8 text-amber-500 flex-shrink-0" />
          <div className="text-center sm:text-left">
            <h3 className="text-amber-500 font-bold text-lg mb-1">Medical Disclaimer</h3>
            <p className="text-sm leading-relaxed">
              This tool is for research and informational purposes only. It is not intended to replace professional medical diagnosis. Always consult a clinical health care provider for clinical purposes.
            </p>
          </div>
        </div>
      </div>

      {/* Fake Alert Modal */}
      {fakeAlert && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md animate-in fade-in duration-300 overflow-y-auto">
          <div className="bg-slate-900 border border-red-500/50 rounded-3xl shadow-[0_0_50px_rgba(239,68,68,0.2)] max-w-2xl w-full p-8 relative animate-in zoom-in-95 duration-300 my-8">
            {/* Pulsing red background elements */}
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-red-500 via-rose-500 to-red-500 animate-pulse"></div>
            <div className="absolute -top-32 -right-32 w-64 h-64 bg-red-500/20 rounded-full blur-[80px] pointer-events-none"></div>
            
            <button 
              onClick={() => setFakeAlert(null)}
              className="absolute top-4 right-4 text-slate-400 hover:text-white transition-colors"
            >
              <X className="w-6 h-6" />
            </button>

            <div className="flex flex-col items-center text-center">
              <div className="w-20 h-20 bg-red-500/10 border border-red-500/30 rounded-full flex items-center justify-center mb-6 shadow-[0_0_20px_rgba(239,68,68,0.4)]">
                <AlertTriangle className="w-10 h-10 text-red-500 drop-shadow-[0_0_10px_rgba(239,68,68,0.8)]" />
              </div>
              
              <h3 className="text-3xl font-bold text-white mb-3 tracking-tight">
                Authenticity Failed
              </h3>
              
              <p className="text-rose-200 text-lg mb-6 leading-relaxed">
                Analysis halted. Our deepfake detection model identified this image as 
                AI-generated with <span className="font-bold text-red-400 bg-red-500/10 px-2 py-0.5 rounded">{(fakeAlert.auth_probability * 100).toFixed(1)}%</span> confidence.
              </p>

              {fakeAlert.explanation_image && (
                <div className="w-full bg-slate-950/50 rounded-2xl p-4 md:p-6 border border-slate-800 mb-8 relative">
                   <h4 className="text-slate-300 text-sm font-semibold uppercase tracking-wider mb-4 pb-2 border-b border-slate-800 text-left">
                     AI Explanation Heatmap
                   </h4>
                   <div className="flex flex-col md:flex-row gap-4 items-center justify-center">
                      <div className="flex-1 w-full max-w-[200px] md:max-w-xs">
                        <img src={imagePreview} alt="Original Upload" className="w-full h-auto rounded-xl border border-slate-700 aspect-square object-cover" />
                        <p className="text-xs text-slate-500 mt-2 font-medium">Original X-Ray</p>
                      </div>
                      <div className="flex-1 w-full max-w-[200px] md:max-w-xs">
                        <img src={fakeAlert.explanation_image} alt="Explainability Heatmap" className="w-full h-auto rounded-xl border border-red-500/50 shadow-[0_0_15px_rgba(239,68,68,0.2)] aspect-square object-cover" />
                        <p className="text-xs text-red-400 mt-2 font-medium">Suspicious Artifacts Highlighted (Heatmap)</p>
                      </div>
                   </div>
                   <p className="text-xs text-slate-400 mt-4 text-left leading-relaxed">
                     The Grad-CAM heatmap highlights areas the AI model focused on. Bright red/yellow regions indicate unnatural patterns or synthetic artifacts typical of GAN-generated medical images.
                   </p>
                </div>
              )}

              <button 
                onClick={() => setFakeAlert(null)}
                className="w-full py-4 bg-red-600 hover:bg-red-700 text-white rounded-xl font-bold shadow-lg shadow-red-600/20 transition-all hover:scale-[1.02]"
              >
                Acknowledge & Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Rejection Modal */}
      {rejectedAlert && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md animate-in fade-in duration-300 overflow-y-auto">
          <div className="bg-slate-900 border border-amber-500/50 rounded-3xl shadow-[0_0_50px_rgba(245,158,11,0.2)] max-w-lg w-full p-8 relative animate-in zoom-in-95 duration-300 my-8">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-amber-500 via-yellow-500 to-amber-500 animate-pulse"></div>
            <div className="absolute -top-32 -right-32 w-64 h-64 bg-amber-500/20 rounded-full blur-[80px] pointer-events-none"></div>
            
            <button 
              onClick={() => setRejectedAlert(null)}
              className="absolute top-4 right-4 text-slate-400 hover:text-white transition-colors"
            >
              <X className="w-6 h-6" />
            </button>

            <div className="flex flex-col items-center text-center">
              <div className="w-20 h-20 bg-amber-500/10 border border-amber-500/30 rounded-full flex items-center justify-center mb-6 shadow-[0_0_20px_rgba(245,158,11,0.4)]">
                <ShieldAlert className="w-10 h-10 text-amber-500 drop-shadow-[0_0_10px_rgba(245,158,11,0.8)]" />
              </div>
              
              <h3 className="text-3xl font-bold text-white mb-3 tracking-tight">
                {rejectedAlert.isGeneralRejection ? "Analysis Done" : "Not a Chest X-Ray"}
              </h3>
              
              {rejectedAlert.isGeneralRejection ? (
                <div className="text-left w-full mb-6 space-y-4">
                  <p className="text-amber-200 text-sm leading-relaxed text-center mb-4">
                    {rejectedAlert.reason || "We could not process your request because both input pipelines encountered errors:"}
                  </p>
                  
                  <div className="bg-slate-950/50 rounded-xl p-4 border border-slate-800">
                    <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-2">
                       📸 Image Pipeline
                    </h4>
                    <p className="text-sm text-rose-300">{rejectedAlert.imageError}</p>
                  </div>
                  
                  <div className="bg-slate-950/50 rounded-xl p-4 border border-slate-800">
                    <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-2">
                       📄 Report Pipeline
                    </h4>
                    <p className="text-sm text-amber-300">{rejectedAlert.reportError}</p>
                  </div>
                </div>
              ) : (
                <>
                  <p className="text-amber-200 text-lg mb-3 leading-relaxed">
                    {rejectedAlert.reason}
                  </p>
                  <p className="text-slate-400 text-sm mb-8">
                    BiomedCLIP confidence: <span className="font-bold text-amber-400">{(rejectedAlert.confidence * 100).toFixed(1)}%</span>
                  </p>
                </>
              )}

              {imagePreview && !rejectedAlert.isGeneralRejection && (
                <div className="w-full max-w-[200px] mb-8">
                  <img src={imagePreview} alt="Uploaded" className="w-full h-auto rounded-xl border border-amber-500/30 aspect-square object-cover" />
                </div>
              )}

              <button 
                onClick={() => setRejectedAlert(null)}
                className="w-full py-4 bg-amber-600 hover:bg-amber-700 text-white rounded-xl font-bold shadow-lg shadow-amber-600/20 transition-all hover:scale-[1.02] mt-4"
              >
                Acknowledge
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
