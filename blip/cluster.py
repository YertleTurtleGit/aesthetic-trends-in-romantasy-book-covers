import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('omw-1.4')

def get_synonyms(word):
    """Get all synonyms for a word using WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word, pos=wordnet.NOUN):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower())
    return synonyms

def are_synonyms(word1, word2):
    """Check if two words are synonyms."""
    word1_synonyms = get_synonyms(word1)
    return word2.lower() in word1_synonyms

def group_similar_words(words):
    """Group similar words together and return a representative for each group."""
    if not words:
        return []
    
    # Convert to lowercase for comparison
    words = [w.lower() for w in words]
    
    # Remove simple duplicates first
    unique_words = list(dict.fromkeys(words))
    
    # Group synonyms
    final_words = []
    used_words = set()
    
    for word in unique_words:
        if word in used_words:
            continue
            
        # Find all synonyms of current word
        current_group = {word}
        for other_word in unique_words:
            if other_word != word and other_word not in used_words:
                if are_synonyms(word, other_word):
                    current_group.add(other_word)
        
        # Add all words in group to used_words
        used_words.update(current_group)
        
        # Use the most common word from the group
        # (or the first one if they're equally common)
        group_list = list(current_group)
        word_frequencies = {w: words.count(w) for w in group_list}
        most_common = max(group_list, key=lambda w: word_frequencies[w])
        final_words.append(most_common)
    
    return final_words


def load_and_preprocess_data(json_file, csv_file):
    """Load JSON data and CSV data, extract clean BLIP descriptions and nouns."""
    # Load JSON data
    with open(json_file) as f:
        data = json.load(f)
    
    # Load CSV data
    df = pd.read_csv(csv_file)

    # Convert "Release Date" to datetime and extract year
    df["Release Date"] = pd.to_datetime(df["Release Date"], errors="coerce")
    df["year"] = df["Release Date"].dt.year

    # Filter for books from 2005 onwards
    df = df[df["year"] >= 2005]
    
    # Define custom stopwords
    custom_stopwords = {'saga','cover', 'book', 'image', 'picture', 'main', 'object', 'words', 'photo', 'front', 'middle', 'right', 'title', 'books', 'background', 'poster','movie', 'scene', 'novel', 'center', 'colors'}
    # Add NLTK English stopwords to the custom stopwords
    stop_words = set(stopwords.words('english')).union(custom_stopwords).union(stopwords.words('german'))
    
    # Define phrases that indicate an invalid description
    # Happend sometimes with the second description; but sometimes the second one worked much better than the first one
    # So i decided to keep it inside and just filter out the invalid ones
    invalid_phrases = [
        "is the cover of the book",
        "is the cover of a book",
        "is the cover of a novel"
    ]
    
    # Extract and clean descriptions, keeping only nouns
    clean_descriptions = []
    image_ids = []
    
    for image_id, image_data in data.items():
        descriptions = image_data['blip_descriptions'][:2]
        
        # Clean and validate descriptions
        valid_descriptions = []
        for desc in descriptions:
            # First remove the prefix
            cleaned = desc.replace('the main object on the image is ', '').strip()
            
            # Check if the cleaned description contains any invalid phrases
            is_valid = True
            for phrase in invalid_phrases:
                if phrase.lower() in cleaned.lower():
                    is_valid = False
                    break
            if is_valid:
                valid_descriptions.append(desc)
        
        # If we have no valid descriptions, use only the first description
        if not valid_descriptions and descriptions:
            valid_descriptions = [descriptions[0]]
        
        # Collect nouns from valid descriptions
        all_nouns = []
        for desc in valid_descriptions:
            # Clean description
            cleaned = desc.replace('a romantic fantasy cover featuring ', '')\
                        .replace('the main object on the image is ', '')\
                        .strip()
            
            # Remove custom stopwords
            tokens = word_tokenize(cleaned)
            filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
            cleaned = ' '.join(filtered_tokens)
            
            # Extract nouns
            pos_tags = pos_tag(word_tokenize(cleaned))
            nouns = [word for word, pos in pos_tags if pos.startswith('NN')]
            all_nouns.extend(nouns)
        
        if all_nouns:  # Only add if we found nouns
            # Group similar words and remove duplicates
            unique_nouns = group_similar_words(all_nouns)
            if unique_nouns:  # Check again if we still have words after grouping
                clean_descriptions.append(' '.join(unique_nouns))
                image_ids.append(image_id.replace('.jpg', ''))
    
    # Create DataFrame with descriptions and image IDs
    desc_df = pd.DataFrame({
        'description': clean_descriptions,
        'image_id': image_ids
    })

    desc_df['image_id'] = desc_df['image_id'].astype(str)
    df['ID'] = df['ID'].astype(str)
    
    # Merge with original CSV data
    merged_df = desc_df.merge(df, left_on='image_id', right_on='ID', how='inner')
    merged_df = merged_df[merged_df["year"] >= 2010]
    
    return merged_df

def create_embeddings(descriptions):
    """Creates BERT embeddings for the descriptions"""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(descriptions)
    return embeddings

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def find_optimal_clusters(embeddings, max_clusters=20):
    """Finding optimal number of clusters using silhouette score"""
    # Normalize the embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=50, max_iter=300)
        cluster_labels = kmeans.fit_predict(embeddings)
        silhouette_avg = silhouette_score(embeddings_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Score vs Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.savefig('silhouette_scores.jpg')
    plt.close()
    
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    return optimal_clusters

def cluster_descriptions(embeddings, n_clusters):
    """K-means clustering"""
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init= 50, max_iter = 300)
    return kmeans.fit_predict(embeddings_scaled)

def get_cluster_representative_terms(embeddings, clusters, descriptions, n_terms=5):
    """
    Identifys the most representative terms for each cluster using cosine similarity
    between term embeddings and cluster centroids.
    """
    
    # Normalize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Calculate cluster centroids
    n_clusters = len(set(clusters))
    centroids = np.zeros((n_clusters, embeddings.shape[1]))
    for i in range(n_clusters):
        centroids[i] = embeddings_scaled[clusters == i].mean(axis=0)
    
    # Get unique terms from all descriptions
    all_terms = set()
    for desc in descriptions:
        all_terms.update(desc.split())
    
    # Create embeddings for individual terms
    model = SentenceTransformer('all-MiniLM-L6-v2')
    term_embeddings = model.encode(list(all_terms))
    
    # Calculate similarity between terms and centroids
    term_importance = {}
    for cluster_idx in range(n_clusters):
        similarities = cosine_similarity(term_embeddings, centroids[cluster_idx].reshape(1, -1)).flatten()
        term_scores = list(zip(all_terms, similarities))
        term_scores.sort(key=lambda x: x[1], reverse=True)
        term_importance[cluster_idx] = term_scores[:n_terms]
    
    return term_importance

def create_cluster_year_visualization(df, clusters, term_importance, cluster_group_size=1):
    """Create line charts with representative terms and unique years per cluster."""
    df_viz = df.copy()
    df_viz['Cluster'] = clusters
    
    # Get total books per year
    total_books_per_year = df_viz.groupby('year').size()
    cluster_year_counts = df_viz.groupby(['Cluster', 'year']).size().unstack(fill_value=0)
    normalized_counts = cluster_year_counts.div(total_books_per_year, axis=1)
    
    # Replace 0s with NaN to handle missing values
    normalized_counts = normalized_counts.replace(0, np.nan)
    
    # Get unique years per cluster
    cluster_years = {}
    for cluster in range(max(clusters) + 1):
        years = sorted(df_viz[df_viz['Cluster'] == cluster]['year'].unique())
        cluster_years[cluster] = years
    
    cluster_list = sorted(normalized_counts.index)
    cluster_groups = [cluster_list[i:i + cluster_group_size] for i in range(0, len(cluster_list), cluster_group_size)]
    
    for group_idx, cluster_group in enumerate(cluster_groups):
        plt.figure(figsize=(15, 8))
        
        for cluster in cluster_group:
            cluster_data = normalized_counts.loc[cluster]
            
            # Connect only existing points
            mask = ~np.isnan(cluster_data)
            x = cluster_data.index[mask]
            y = cluster_data[mask]
            
            # Plot lines and points
            plt.plot(x, y, '-o', markersize=6, markerfacecolor='white', 
                    markeredgewidth=2, linewidth=2)
            
            # Get representative terms
            terms = [f"{term} ({score:.2f})" for term, score in term_importance[cluster]]
            terms_str = ", ".join(terms)
            
            # Get years for this cluster
            years = cluster_years[cluster]
            years_str = f"Years: {', '.join(map(str, years))}"
            
            plt.plot([], [], 
                    label=f'Cluster {cluster}\n[{terms_str}]\n{years_str}')
        
        if len(cluster_group) > 1:
            title = f'Normalized Year Distribution for Clusters {cluster_group[0]} to {cluster_group[-1]}'
        else:
            title = f'Normalized Year Distribution for Cluster {cluster_group[0]}'
            
        plt.title(title, fontsize=14)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Normalized Proportion of Books', fontsize=12)
        plt.legend(title='Cluster & Representative Terms', 
                  bbox_to_anchor=(1.05, 1), 
                  loc='upper left', 
                  fontsize=8,
                  borderaxespad=0.)
        plt.grid(alpha=0.3)
        
        # Set y-axis limits with some padding
        plt.ylim(0, normalized_counts.max().max() * 1.1)
        
        # Ensure x-axis shows all years
        plt.xticks(sorted(normalized_counts.columns), rotation=45)
        
        plt.tight_layout()
        
        plt.savefig(f'cluster_{group_idx}_direct.jpg', 
                   bbox_inches='tight', 
                   dpi=300)
        plt.close()

def analyze_clusters(df, clusters):
    """Analyze the contents of each cluster with temporal information."""
    cluster_analysis = {}
    
    for cluster_id in range(max(clusters) + 1):
        cluster_data = df[clusters == cluster_id]
        
        cluster_analysis[cluster_id] = {
            'size': len(cluster_data),
            'year_range': f"{cluster_data['year'].min()} - {cluster_data['year'].max()}",
            'mean_year': cluster_data['year'].mean(),
            'sample_descriptions': cluster_data['description'].head().tolist(),
            'most_common_years': cluster_data['year'].value_counts().head().to_dict()
        }
    
    return cluster_analysis

def main():
    
    print("Loading and processing data...")
    df = load_and_preprocess_data('images/enhanced_analysis_results.json', 'final_books.csv') 
    # enhanced_analysis_results_new.json was a later running with 3 Descriptions per image
    # For the run of the results, use enhanced_analysis_results.json
    # Code is also working with the enhanced_analysis_results_new.json but gives maybe different results cause of the randomness (temperature parameter in blib_descriptions.py) of BLIP descriptions
    
    print(f"Number of books after filtering (>= 2010): {len(df)}")  
    
    # Create embeddings
    print("Creating BERT embeddings...")
    embeddings = create_embeddings(df['description'].tolist())
    
    # Find optimal number of clusters
    print("Finding optimal number of clusters...")
    optimal_clusters = find_optimal_clusters(embeddings)
    print(f"Optimal number of clusters: {optimal_clusters}")
    
    # Cluster the descriptions
    print("Clustering descriptions...")
    clusters = cluster_descriptions(embeddings, optimal_clusters)
    
    # Get representative terms for each cluster
    print("Identifying representative terms for each cluster...")
    term_importance = get_cluster_representative_terms(embeddings, clusters, df['description'].tolist())
    
    # Create line plot visualizations
    print("Creating year distribution visualization...")
    create_cluster_year_visualization(df, clusters, term_importance)
    
    # Analyze clusters
    cluster_analysis = analyze_clusters(df, clusters)

    # Print results
    print("\nCluster Analysis:")
    for cluster_id, analysis in cluster_analysis.items():
        print(f"\nCluster {cluster_id}:")
        print(f"Size: {analysis['size']} descriptions")
        print(f"Year range: {analysis['year_range']}")
        print(f"Mean year: {analysis['mean_year']:.2f}")
        print("Most common years:")
        for year, count in analysis['most_common_years'].items():
            print(f"  {year}: {count} books")
        print("Sample descriptions:")
        for desc in analysis['sample_descriptions']:
            print(f"- {desc}")

if __name__ == "__main__":
    main()