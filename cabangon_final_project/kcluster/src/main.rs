use csv::ReaderBuilder;
use linfa::prelude::*;
use linfa_clustering::KMeans;
use ndarray::Array2;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Define the path to the obesity CSV file
    let file_path = "ObesityDataSet_raw_and_data_sinthetic.csv";

    // Define the numeric features to use (the 8 listed in the proposal)
    let numeric_features = vec![
        "Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE",
    ];

    // Opens the CSV file
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(file_path)?;

    // Read headers of the file
    let headers = rdr.headers()?.clone();

    // Map feature names to their column indices
    let feature_indices: Vec<usize> = numeric_features
        .iter()
        .map(|feature| {
            headers
                .iter()
                .position(|h| h == *feature)
                .expect(&format!("Feature {} not found in headers", feature))
        })
        .collect();

    // Also get the index of the label column
    let label_index = headers
        .iter()
        .position(|h| h == "NObeyesdad")
        .expect("Label column not found");

    // Prepare data structures
    let mut data_matrix: Vec<f64> = Vec::new();
    let mut labels: Vec<String> = Vec::new();
    let mut row_count = 0;

    // Read and process each record
    for result in rdr.records() {
        let record = result?;
        let mut row = Vec::new();

        // Extract numeric features of the rows
        for &idx in &feature_indices {
            let val = record.get(idx).unwrap_or("0").parse::<f64>().unwrap_or(0.0);
            row.push(val);
        }

        if row.len() == numeric_features.len() {
            data_matrix.extend(row);
            row_count += 1;

            // Extract label
            let label = record.get(label_index).unwrap_or("").to_string();
            labels.push(label);
        }
    }

    // Create an ndarray from the data
    let array: Array2<f64> =
        Array2::from_shape_vec((row_count, numeric_features.len()), data_matrix)?;

    // Create a dataset
    let dataset = DatasetBase::from(array.clone());

    // Define the number of clusters
    let n_clusters = 3;

    // Fit KMeans model
    let model = KMeans::params(n_clusters).fit(&dataset)?;

    // Predict cluster assignments
    let cluster_assignments = model.predict(&dataset);

    // Print the cluster sizes
    println!("Cluster distribution:");
    let mut cluster_counts = vec![0; n_clusters];
    for &c in cluster_assignments.iter() {
        cluster_counts[c] += 1;
    }
    for (i, count) in cluster_counts.iter().enumerate() {
        println!("Cluster {}: {} rows", i, count);
    }
    // Computes and Prints the Cluster Distance
    let model = KMeans::params(n_clusters).fit(&dataset)?;
    let centroids = model.centroids(); //
    let cluster_assignments = model.predict(&dataset);
    let mut distances_by_cluster = vec![Vec::new(); n_clusters];
    for (i, row) in array.outer_iter().enumerate() {
        let cluster = cluster_assignments[i];
        let centroid = centroids.row(cluster);

        let dist = row
            .iter()
            .zip(centroid.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        distances_by_cluster[cluster].push(dist);
    }

    println!("\n📏 Cluster Distance Stats (to centroid):");
    for (cluster_id, dists) in distances_by_cluster.iter().enumerate() {
        let count = dists.len() as f64;
        let mean_dist = dists.iter().sum::<f64>() / count;
        let max_dist = dists.iter().cloned().fold(f64::MIN, f64::max);
        let min_dist = dists.iter().cloned().fold(f64::MAX, f64::min);

        println!(
            "Cluster {} → Avg Distance: {:.4}, Min: {:.4}, Max: {:.4}",
            cluster_id, mean_dist, min_dist, max_dist
        );
    }

    // Print the average of the 8 feature values per cluster
    println!("Average Feature Values per Cluster:");
    for cluster_id in 0..n_clusters {
        let mask = cluster_assignments
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c == cluster_id);
        let mut feature_sums = vec![0.0; numeric_features.len()];
        let mut count = 0;

        for (i, _) in mask {
            for (j, val) in array.row(i).iter().enumerate() {
                feature_sums[j] += val;
            }
            count += 1;
        }

        if count > 0 {
            println!("\nCluster {}:", cluster_id);
            for (i, &sum) in feature_sums.iter().enumerate() {
                let mean = sum / count as f64;
                println!("{:<10} = {:.2}", numeric_features[i], mean);
            }
        }
    }

    // This part identifies representative individuals (which ones are closest to centroid)
    println!("\n🌟 Representative Individuals per Cluster:");
    for cluster_id in 0..n_clusters {
        let centroid = model.centroids().row(cluster_id);
        let mut min_distance = f64::MAX;
        let mut representative_index = 0;

        for (i, row) in array.outer_iter().enumerate() {
            if cluster_assignments[i] == cluster_id {
                let distance = row
                    .iter()
                    .zip(centroid.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if distance < min_distance {
                    min_distance = distance;
                    representative_index = i;
                }
            }
        }

        println!(
            "Cluster {}: Representative Index = {}, Label = {}",
            cluster_id, representative_index, labels[representative_index]
        );
    }

    Ok(())
}
