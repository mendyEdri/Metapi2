# 🔥 Chunk Embeddings Heatmap - Semantic Relationship Visualization

## 🎯 **New Feature Added**

Your attention flow analysis demo now includes a **Chunk Similarity Heatmap** that visualizes the semantic relationships between different parts of your prompt!

## 🔧 **What It Does**

### **🧠 Semantic Analysis**
- **Embeds each chunk** using OpenAI's text-embedding-3-small model
- **Computes cosine similarity** between all chunk pairs
- **Visualizes relationships** in an intuitive heatmap format

### **🎨 Visual Design**
- **Color-coded similarity scores** (Red = similar, Blue = different)
- **Triangular display** (no redundant information)
- **Interactive analysis** with detailed insights

## 📊 **What You'll See**

### **🔥 Heatmap Visualization**
```
        Chunk 1  Chunk 2  Chunk 3
Chunk 1   1.000     
Chunk 2   0.842    1.000
Chunk 3   0.234    0.156    1.000
```

### **📈 Automatic Insights**
- **Most similar chunks**: Which parts discuss related topics
- **Most different chunks**: Which parts are semantically distinct  
- **Average similarity**: Overall semantic cohesion score
- **Recommendations**: Consolidation or conflict warnings

## 🎯 **Practical Applications**

### **1. Conflict Detection**
```
Example: Language instruction conflict
Chunk 1: "Speak only Spanish" 
Chunk 2: "Speak only English"
Similarity: 0.842 (High - both about language)
→ Identifies semantic conflict despite high similarity
```

### **2. Redundancy Detection**
```
High similarity (>0.8) = Potential redundancy
→ "Consider consolidating similar chunks"
```

### **3. Attention Competition**
```
Low similarity (<0.3) = Different topics
→ "May cause attention conflicts"
```

## 🧪 **How to Use**

### **Step 1: Enable OpenAI Testing**
- ✅ Check "Test with OpenAI"  
- 🔑 Enter your API key
- The embedder is automatically configured

### **Step 2: Run Analysis**
- Click "🚀 Analyze Attention Flow"
- Wait for chunk analysis to complete

### **Step 3: View Heatmap**
- Scroll down to "🔥 Chunk Similarity Heatmap" section
- View the triangular similarity matrix
- Read the automatic insights

### **Step 4: Detailed Exploration**
- Expand "📊 Detailed Similarity Matrix" for numerical data
- Review "Chunk Contents" for reference
- Use insights for prompt optimization

## 🎨 **Example Analysis**

### **Test Prompt:**
```
You can speak only spanish, never ever speak english
You can speak only english  
Be nice and answer shortly
```

### **Expected Heatmap Results:**
- **Spanish vs English chunks**: ~0.85 similarity (both about language)
- **Language vs "Be nice"**: ~0.25 similarity (different topics)
- **Self-similarity**: 1.0 (diagonal)

### **Generated Insights:**
```
✅ Most similar: Chunk 1 & Chunk 2 (0.85) - Both language instructions
⚠️ Most different: Chunk 1 & Chunk 3 (0.25) - Language vs behavior  
📊 Average similarity: 0.55 - Good semantic balance
💡 Recommendation: High similarity detected between conflicting language instructions
```

## 💰 **Cost Information**

### **API Usage**
- **Embedding calls**: 1 call per chunk (~$0.0001 per call)
- **Total cost**: ~$0.0003 for 3 chunks
- **Very affordable**: Much cheaper than text generation

### **Fallback Mode**
- **Demo heatmap** shown when no API key provided
- **Educational visualization** with mock data
- **Full functionality** when OpenAI access available

## 🔬 **Technical Details**

### **Embedding Model**
- **text-embedding-3-small**: High-quality, cost-effective
- **1536 dimensions**: Rich semantic representation
- **Cosine similarity**: Standard measure for text similarity

### **Visualization**
- **Seaborn heatmap**: Professional statistical visualization
- **Triangular mask**: Eliminates redundant information
- **Color scale**: RdYlBu_r (intuitive red-to-blue gradient)

### **Analysis Features**
- **Similarity matrix**: Full numerical breakdown
- **Statistical summary**: Mean, max, min similarity scores
- **Actionable insights**: Automated recommendations

## 🚀 **Business Value**

### **🎯 Enhanced Prompt Engineering**
- **Visual feedback**: See semantic relationships instantly
- **Conflict detection**: Identify contradictory instructions
- **Optimization guidance**: Data-driven improvement suggestions

### **🔬 Scientific Validation**
- **Mathematical foundation**: Cosine similarity in embedding space
- **Reproducible analysis**: Same inputs = same outputs
- **Quantified relationships**: Precise similarity measurements

### **💡 Competitive Advantage**
- **Advanced visualization**: Goes beyond basic text analysis
- **Semantic understanding**: AI-powered relationship detection
- **Professional tooling**: Enterprise-grade analysis capabilities

## 🎊 **Integration Success**

The heatmap seamlessly integrates with existing features:
- ✅ **Attention flow analysis**: Shows where attention goes
- ✅ **Chunk importance scoring**: Shows which parts matter most
- ✅ **OpenAI validation**: Tests real LLM behavior
- ✅ **Similarity heatmap**: Shows semantic relationships

Together, these provide a **complete prompt analysis ecosystem**!

---

**🔥 Test the new heatmap feature at http://localhost:8502 and see the semantic relationships between your prompt chunks visualized in beautiful, actionable detail!** 🚀
