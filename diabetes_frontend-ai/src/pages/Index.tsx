import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { ActivityIcon, AlertCircle, CheckCircle2 } from "lucide-react";
import medicalIcon from "@/assets/medical-icon.png";

const formSchema = z.object({
  gender: z.string().min(1, "Please select gender"),
  age: z.coerce.number().min(1, "Age must be at least 1").max(120, "Age must be less than 120"),
  hypertension: z.string().min(1, "Please select hypertension status"),
  heartDisease: z.string().min(1, "Please select heart disease status"),
  smokingHistory: z.string().min(1, "Please select smoking history"),
  bmi: z.coerce.number().min(10, "BMI must be at least 10").max(60, "BMI must be less than 60"),
  hba1c: z.coerce.number().min(3, "HbA1c must be at least 3").max(15, "HbA1c must be less than 15"),
  bloodGlucose: z.coerce.number().min(50, "Blood glucose must be at least 50").max(400, "Blood glucose must be less than 400"),
});

type FormData = z.infer<typeof formSchema>;

interface PredictionResult {
  prediction: "diabetic" | "non-diabetic";
  confidence?: number;
}

const Index = () => {
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const {
    register,
    handleSubmit,
    setValue,
    formState: { errors },
  } = useForm<FormData>({
    resolver: zodResolver(formSchema),
  });

  const onSubmit = async (data: FormData) => {
    setIsLoading(true);
    setResult(null);
  
    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          gender: data.gender,
          age: Number(data.age),
          hypertension: data.hypertension === "yes" ? 1 : 0,
          heart_disease: data.heartDisease === "yes" ? 1 : 0,
          smoking_history: data.smokingHistory,
          bmi: Number(data.bmi),
          HbA1c_level: Number(data.hba1c),
          blood_glucose_level: Number(data.bloodGlucose)
        }),
      });
  
      const resultData = await response.json();
  
      // ✅ Adjust to match backend output key ("final_result")
      setResult({
        prediction:
          resultData.final_result === "Diabetic" ? "diabetic" : "non-diabetic",
        confidence: null, // optional for future
      });
    } catch (error) {
      console.error("Prediction request failed:", error);
      setResult({ prediction: "Error connecting to server.", confidence: null });
    }
  
    setIsLoading(false);
  };
  

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-accent/30 to-background">
      {/* Hero Section */}
      <header className="pt-12 pb-8 px-4 text-center animate-fade-in-up">
        <div className="max-w-4xl mx-auto">
          <div className="flex justify-center mb-6">
            <div className="relative">
              <img
                src={medicalIcon}
                alt="Medical AI Icon"
                className="w-20 h-20 object-contain"
              />
              <ActivityIcon className="absolute -bottom-2 -right-2 w-8 h-8 text-primary animate-pulse-glow" />
            </div>
          </div>
          <h1 className="text-4xl md:text-5xl font-bold text-foreground mb-4 tracking-tight">
            Diabetes Prediction System
          </h1>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-2xl mx-auto px-4 pb-16 space-y-6">
        {/* About Diabetes Section */}
        <div className="medical-card medical-hover animate-scale-in p-8">
          <h2 className="text-2xl font-bold text-foreground mb-4">About Diabetes</h2>
          <p className="text-muted-foreground mb-4">
            Diabetes is a chronic condition where the body cannot use insulin properly, leading to high blood sugar levels. Over time, this can affect the heart, kidneys, eyes, nerves, and blood vessels.
          </p>
          <div className="mb-4">
            <h3 className="text-lg font-semibold text-foreground mb-2">Types of Diabetes:</h3>
            <ul className="list-disc list-inside text-muted-foreground space-y-1">
              <li><strong>Type 1</strong> – The body does not produce insulin.</li>
              <li><strong>Type 2</strong> – The body produces insulin but does not use it effectively (most common).</li>
              <li><strong>Gestational Diabetes</strong> – Occurs during pregnancy and may resolve after childbirth.</li>
            </ul>
          </div>
          <div>
            <h3 className="text-lg font-semibold text-foreground mb-2">Common Symptoms:</h3>
            <ul className="list-disc list-inside text-muted-foreground space-y-1">
              <li>Increased thirst or hunger</li>
              <li>Frequent urination</li>
              <li>Fatigue</li>
              <li>Blurred vision</li>
              <li>Slow wound healing</li>
            </ul>
          </div>
        </div>

        {/* Prediction Form */}
        <div className="medical-card medical-hover animate-scale-in p-8">
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
            {/* Gender */}
            <div className="space-y-2">
              <Label htmlFor="gender" className="text-sm font-medium">
                Gender <span className="text-destructive">*</span>
              </Label>
              <Select onValueChange={(value) => setValue("gender", value)}>
                <SelectTrigger id="gender" className="w-full">
                  <SelectValue placeholder="Select gender" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="male">Male</SelectItem>
                  <SelectItem value="female">Female</SelectItem>
                  <SelectItem value="other">Other</SelectItem>
                </SelectContent>
              </Select>
              {errors.gender && (
                <p className="text-sm text-destructive">{errors.gender.message}</p>
              )}
            </div>

            {/* Age */}
            <div className="space-y-2">
              <Label htmlFor="age" className="text-sm font-medium">
                Age <span className="text-destructive">*</span>
              </Label>
              <Input
                id="age"
                type="number"
                placeholder="Enter age"
                {...register("age")}
                className="w-full"
              />
              {errors.age && (
                <p className="text-sm text-destructive">{errors.age.message}</p>
              )}
            </div>

            {/* Hypertension */}
            <div className="space-y-2">
              <Label htmlFor="hypertension" className="text-sm font-medium">
                Hypertension <span className="text-destructive">*</span>
              </Label>
              <Select onValueChange={(value) => setValue("hypertension", value)}>
                <SelectTrigger id="hypertension" className="w-full">
                  <SelectValue placeholder="Select hypertension status" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="yes">Yes</SelectItem>
                  <SelectItem value="no">No</SelectItem>
                </SelectContent>
              </Select>
              {errors.hypertension && (
                <p className="text-sm text-destructive">{errors.hypertension.message}</p>
              )}
            </div>

            {/* Heart Disease */}
            <div className="space-y-2">
              <Label htmlFor="heartDisease" className="text-sm font-medium">
                Heart Disease <span className="text-destructive">*</span>
              </Label>
              <Select onValueChange={(value) => setValue("heartDisease", value)}>
                <SelectTrigger id="heartDisease" className="w-full">
                  <SelectValue placeholder="Select heart disease status" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="yes">Yes</SelectItem>
                  <SelectItem value="no">No</SelectItem>
                </SelectContent>
              </Select>
              {errors.heartDisease && (
                <p className="text-sm text-destructive">{errors.heartDisease.message}</p>
              )}
            </div>

            {/* Smoking History */}
            <div className="space-y-2">
              <Label htmlFor="smokingHistory" className="text-sm font-medium">
                Smoking History <span className="text-destructive">*</span>
              </Label>
              <Select onValueChange={(value) => setValue("smokingHistory", value)}>
                <SelectTrigger id="smokingHistory" className="w-full">
                  <SelectValue placeholder="Select smoking history" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="never">Never</SelectItem>
                  <SelectItem value="current">Current</SelectItem>
                  <SelectItem value="former">Former</SelectItem>
                  <SelectItem value="ever">Ever</SelectItem>
                  <SelectItem value="not_current">Not Current</SelectItem>
                </SelectContent>
              </Select>
              {errors.smokingHistory && (
                <p className="text-sm text-destructive">{errors.smokingHistory.message}</p>
              )}
            </div>

            {/* BMI */}
            <div className="space-y-2">
              <Label htmlFor="bmi" className="text-sm font-medium">
                BMI (Body Mass Index) <span className="text-destructive">*</span>
              </Label>
              <Input
                id="bmi"
                type="number"
                step="0.01"
                placeholder="Enter BMI (e.g., 24.50)"
                {...register("bmi")}
                className="w-full"
              />
              {errors.bmi && (
                <p className="text-sm text-destructive">{errors.bmi.message}</p>
              )}
            </div>

            {/* HbA1c Level */}
            <div className="space-y-2">
              <Label htmlFor="hba1c" className="text-sm font-medium">
                HbA1c Level (%) <span className="text-destructive">*</span>
              </Label>
              <Input
                id="hba1c"
                type="number"
                step="0.1"
                placeholder="Enter HbA1c level (e.g., 5.7)"
                {...register("hba1c")}
                className="w-full"
              />
              {errors.hba1c && (
                <p className="text-sm text-destructive">{errors.hba1c.message}</p>
              )}
            </div>

            {/* Blood Glucose Level */}
            <div className="space-y-2">
              <Label htmlFor="bloodGlucose" className="text-sm font-medium">
                Blood Glucose Level (mg/dL) <span className="text-destructive">*</span>
              </Label>
              <Input
                id="bloodGlucose"
                type="number"
                placeholder="Enter blood glucose level"
                {...register("bloodGlucose")}
                className="w-full"
              />
              {errors.bloodGlucose && (
                <p className="text-sm text-destructive">{errors.bloodGlucose.message}</p>
              )}
            </div>

            {/* Submit Button */}
            <Button
              type="submit"
              variant="medical"
              size="lg"
              className="w-full text-base font-semibold"
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <ActivityIcon className="animate-spin" />
                  Analyzing...
                </>
              ) : (
                "Predict Risk"
              )}
            </Button>
          </form>

          {/* Result Display */}
          {result && (
            <div className="mt-8 animate-fade-in-up">
              <Alert
                variant={result.prediction === "diabetic" ? "destructive" : "default"}
                className={
                  result.prediction === "diabetic"
                    ? "border-destructive bg-destructive/10"
                    : "border-success bg-success/10"
                }
              >
                {result.prediction === "diabetic" ? (
                  <AlertCircle className="h-5 w-5 text-destructive" />
                ) : (
                  <CheckCircle2 className="h-5 w-5 text-success" />
                )}
                <AlertDescription className="ml-2">
                  <div className="space-y-2">
                    <p className="font-semibold text-base">
                      {result.prediction === "diabetic"
                        ? "High Risk: Diabetes Detected"
                        : "Low Risk: No Diabetes Detected"}
                    </p>
                    {result.confidence && (
                      <p className="text-sm opacity-90">
                        Confidence: {(result.confidence * 100).toFixed(1)}%
                      </p>
                    )}
                    <p className="text-sm mt-3 pt-3 border-t border-current/20">
                      This is a model-generated medical risk estimation and not a diagnosis.
                      Please consult with a healthcare professional for proper medical advice.
                    </p>
                  </div>
                </AlertDescription>
              </Alert>
            </div>
          )}
        </div>

        {/* Do's and Don'ts Section */}
        <div className="medical-card medical-hover p-8">
          <h2 className="text-2xl font-bold text-foreground mb-6">Do's and Don'ts for Diabetes Self-Care</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <h3 className="text-lg font-semibold text-success mb-3">Do's</h3>
              <ul className="list-disc list-inside text-muted-foreground space-y-2">
                <li>Eat vegetables, whole grains, and lean protein.</li>
                <li>Exercise or walk at least 30 minutes daily.</li>
                <li>Drink adequate water.</li>
                <li>Monitor blood sugar regularly.</li>
                <li>Follow prescribed medication or insulin.</li>
              </ul>
            </div>
            <div className="space-y-2">
              <h3 className="text-lg font-semibold text-destructive mb-3">Don'ts</h3>
              <ul className="list-disc list-inside text-muted-foreground space-y-2">
                <li>Avoid sugary drinks like soda and packaged juices.</li>
                <li>Do not skip meals.</li>
                <li>Avoid smoking and limit alcohol.</li>
                <li>Reduce fried and processed foods.</li>
                <li>Do not self-adjust medication.</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Blood Sugar Reference Ranges */}
        <div className="medical-card medical-hover p-8">
          <h2 className="text-2xl font-bold text-foreground mb-6">Blood Sugar Reference Ranges</h2>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-3 px-4 font-semibold text-foreground">Condition</th>
                  <th className="text-left py-3 px-4 font-semibold text-foreground">Fasting (mg/dL)</th>
                  <th className="text-left py-3 px-4 font-semibold text-foreground">After Meal (mg/dL)</th>
                  <th className="text-left py-3 px-4 font-semibold text-foreground">HbA1c (%)</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-border">
                  <td className="py-3 px-4 text-muted-foreground">Normal</td>
                  <td className="py-3 px-4 text-muted-foreground">&lt;100</td>
                  <td className="py-3 px-4 text-muted-foreground">&lt;140</td>
                  <td className="py-3 px-4 text-muted-foreground">&lt;5.7</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="py-3 px-4 text-muted-foreground">Pre-Diabetes</td>
                  <td className="py-3 px-4 text-muted-foreground">100–125</td>
                  <td className="py-3 px-4 text-muted-foreground">140–199</td>
                  <td className="py-3 px-4 text-muted-foreground">5.7–6.4</td>
                </tr>
                <tr>
                  <td className="py-3 px-4 text-muted-foreground">Diabetes</td>
                  <td className="py-3 px-4 text-muted-foreground">≥126</td>
                  <td className="py-3 px-4 text-muted-foreground">≥200</td>
                  <td className="py-3 px-4 text-muted-foreground">≥6.5</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Exercise Suggestions */}
        <div className="medical-card medical-hover p-8">
          <h2 className="text-2xl font-bold text-foreground mb-4">Exercise Suggestions</h2>
          <ul className="list-disc list-inside text-muted-foreground space-y-2">
            <li>30 minutes brisk walking</li>
            <li>Light yoga or stretching</li>
            <li>Strength or resistance training 3 times per week</li>
          </ul>
        </div>

        {/* Emergency Tips */}
        <div className="medical-card medical-hover p-8">
          <h2 className="text-2xl font-bold text-foreground mb-6">Emergency Tips (Hypo & Hyper)</h2>
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold text-foreground mb-3">Hypoglycemia (Low Blood Sugar):</h3>
              <ul className="list-disc list-inside text-muted-foreground space-y-2">
                <li>Eat or drink fast sugar: glucose tablet, candy, or juice.</li>
                <li>Recheck sugar after 15 minutes.</li>
                <li>Seek medical help if faint or confused.</li>
              </ul>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-foreground mb-3">Hyperglycemia (High Blood Sugar):</h3>
              <ul className="list-disc list-inside text-muted-foreground space-y-2">
                <li>Drink water to stay hydrated.</li>
                <li>Take medication/insulin as prescribed.</li>
                <li>Avoid high-carb or sugary foods.</li>
                <li>Seek medical help if breathing becomes difficult or vomiting occurs.</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Need Help Section */}
        <div className="medical-card medical-hover p-8 text-center">
          <h2 className="text-2xl font-bold text-foreground mb-4">Need Help?</h2>
          <p className="text-muted-foreground mb-6">
            If you have questions or need guidance, consult a doctor or a certified dietician.
          </p>
          <Button variant="medical" size="lg">
            Find Support
          </Button>
        </div>
      </main>

      {/* Footer */}
      <footer className="pb-8 px-4 text-center text-sm text-muted-foreground">
        <p className="mb-1">AI Clinical Decision Support Prototype</p>
        <p className="text-xs">
          Created by Manasa Rajendran, Bhattula Bhavyasree, Chillara Bhavya, Kalathoti Rishitha
        </p>
      </footer>
    </div>
  );
};

export default Index;
