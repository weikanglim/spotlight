import { ClassLabelResult } from "./class-label-result";
import { Classification } from "./classification";

export class ClassifierResult {
    modelName: string;
    accuracy: number;
    classificationMatrix: ClassLabelResult[];
    classifications: Classification[];
    modelUri: string;
}