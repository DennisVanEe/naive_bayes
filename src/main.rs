mod bayes;

use anyhow::Result;

fn main() -> Result<()> {
    let classifier = bayes::NaiveBayesClassifier::new("D:/Dev/cs145/train.csv")?;
    classifier.predict("D:/Dev/cs145/test.csv", "D:/Dev/cs145/result4.csv")?;

    println!("Done");

    Ok(())
}
