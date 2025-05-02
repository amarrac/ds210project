use csv::ReaderBuilder;
use linfa::prelude::*;
use linfa_clustering::KMeans;
use ndarray::Array2;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let file_path = "ObesityDataSet_raw_and_data_sinthetic.csv";

    // Define numeric features to use
    let numeric_features = vec![
        "Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE",
    ];

    // Read the CSV
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(file_path)?;

    let headers = rdr.headers()?.clone();

    let feature_indices: Vec<usize> = numeric_features
        .iter()
        .map(|f| {
            headers
                .iter()
                .position(|h| h == *f)
                .expect(&format!("Feature {} not found", f))
        })
        .collect();

    let label_index = headers
        .iter()
        .position(|h| h == "NObeyesdad")
        .expect("Label column not found");

    let mut data_matrix: Vec<f64> = Vec::new();
    let mut labels: Vec<String> = Vec::new();
    let mut row_count = 0;

    for result in rdr.records() {
        let record = result?;
        let mut row = Vec::new();

        for &idx in &feature_indices {
            let val = record.get(idx).unwrap_or("0").parse::<f64>().unwrap_or(0.0);
            row.push(val);
        }

        if row.len() == numeric_features.len() {
            data_matrix.extend(row);
            row_count += 1;

            let label = record.get(label_index).unwrap_or("").to_string();
            labels.push(label);
        }
    }

    let array = Array2::from_shape_vec((row_count, numeric_features.len()), data_matrix)?;
    let dataset = DatasetBase::from(array.clone());

    // Fit KMeans with k clusters
    let n_clusters = 3;
    let model = KMeans::params(n_clusters).fit(&dataset)?;
    let cluster_assignments = model.predict(&dataset);

    // Print average feature values per cluster
    println!("Average Feature Values per Cluster:");
    for cluster_id in 0..n_clusters {
        let mut feature_sums = vec![0.0; numeric_features.len()];
        let mut count = 0;

        for (i, &assigned_cluster) in cluster_assignments.iter().enumerate() {
            if assigned_cluster == cluster_id {
                for (j, val) in array.row(i).iter().enumerate() {
                    feature_sums[j] += val;
                }
                count += 1;
            }
        }

        if count > 0 {
            println!("\nCluster {} ({} samples):", cluster_id, count);
            for (i, &sum) in feature_sums.iter().enumerate() {
                println!("{:<10} = {:.2}", numeric_features[i], sum / count as f64);
            }
        }
    }

    Ok(())
}
